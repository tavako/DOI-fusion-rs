#![feature(portable_simd)]
use csv::Writer;
use itertools::Itertools;
use ndarray::{iter::AxisIter, linalg::Dot, prelude::*, stack};
use std::{
    error::Error,
    f64::consts::PI,
    fs::File,
    simd::f64x4,
    time::Instant,
};
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

#[derive(PartialEq)]
pub enum Indexing {
    Xy,
    Ij,
}

pub fn meshgrid<T>(xi: &[Array1<T>], indexing: Indexing) -> Result<Vec<ArrayD<T>>, Box<dyn Error>>
where
    T: Copy,
{
    let ndim = xi.len();
    let product = xi.iter().map(|x| x.iter()).multi_cartesian_product();

    let mut grids: Vec<ArrayD<T>> = Vec::with_capacity(ndim);

    for (dim_index, _) in xi.iter().enumerate() {
        // Generate a flat vector with the correct repeated pattern
        let values: Vec<T> = product.clone().map(|p| *p[dim_index]).collect();

        let mut grid_shape: Vec<usize> = vec![1; ndim];
        grid_shape[dim_index] = xi[dim_index].len();

        // Determine the correct repetition for each dimension
        for (j, len) in xi.iter().map(|x| x.len()).enumerate() {
            if j != dim_index {
                grid_shape[j] = len;
            }
        }

        let grid = Array::from_shape_vec(IxDyn(&grid_shape), values)?;
        grids.push(grid);
    }

    // Swap axes for "xy" indexing
    if matches!(indexing, Indexing::Xy) && ndim > 1 {
        for grid in &mut grids {
            grid.swap_axes(0, 1);
        }
    }

    Ok(grids)
}

fn ndarray_style() {
    let nodes_locs: Array2<f64> = arr2(&[
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
    ]);
    let target_locs: Array2<f64> = arr2(&[[-5.0, -5.0, 10.0], [5.0, 5.0, 5.0], [5.0, 0.0, 10.0]]);

    let mut local_max_vectors: Array3<f64> =
        Array3::zeros((nodes_locs.shape()[0], target_locs.shape()[0], 3));
    for (iter, node_loc) in nodes_locs.axis_iter(Axis(0)).enumerate() {
        local_max_vectors
            .slice_mut(s![iter, .., ..])
            .assign(&(&target_locs - &node_loc));
    }
    //println!("standard version: {} " , local_max_vectors);
    for node_idx in 0..local_max_vectors.shape()[0] {
        for target_idx in 0..local_max_vectors.shape()[1] {
            let norm = local_max_vectors
                .slice(s![node_idx, target_idx, ..])
                .mapv(|x| x.powi(2))
                .sum();
            local_max_vectors
                .slice_mut(s![node_idx, target_idx, ..])
                .mapv_into(|x| x / norm);
        }
    }
    //println!("normalized version: {} " , local_max_vectors);

    let x = Array::linspace(-10.0, 10.0, 256);
    let y = Array::linspace(-10.0, 10.0, 256);
    let z = Array::linspace(0.0, 10.0, 128); // Example with 3D
    let xi = vec![x, y, z];

    let grids: Vec<ArrayBase<ndarray::OwnedRepr<f64>, Dim<ndarray::IxDynImpl>>> =
        meshgrid(&xi, Indexing::Xy).unwrap();
    let fused_grid = stack(
        Axis(3),
        &[grids[0].view(), grids[1].view(), grids[2].view()],
    )
    .unwrap();
    let mut fusion_result = Array3::from_elem(
        (
            fused_grid.shape()[0],
            fused_grid.shape()[1],
            fused_grid.shape()[2],
        ),
        0.0,
    );
    //println!("fused_grid - single_node {:?}" , &fused_grid - &nodes_locs.slice(s![0,..]));
    let mut pointing_space_vector = Array5::from_elem((4, 256, 256, 128, 3), 0.0);
    for (node_iter, node_array) in local_max_vectors.axis_iter(Axis(0)).enumerate() {
        for (_, target_vector_node) in node_array.axis_iter(Axis(0)).enumerate() {
            pointing_space_vector
                .slice_mut(s![node_iter, .., .., .., ..])
                .assign(
                    &(&fused_grid - &nodes_locs.slice(s![node_iter, ..]))
                        .to_shape((256, 256, 128, 3))
                        .unwrap(),
                );
            let mut pointing_space_vector_local =
                pointing_space_vector.slice_mut(s![node_iter, .., .., .., ..]);
            for x in 0..pointing_space_vector_local.shape()[0] {
                for y in 0..pointing_space_vector_local.shape()[1] {
                    for z in 0..pointing_space_vector_local.shape()[2] {
                        let norm = pointing_space_vector_local
                            .slice(s![x, y, z, ..])
                            .mapv(|x| x.powi(2))
                            .sum();
                        pointing_space_vector_local
                            .slice_mut(s![x, y, z, ..])
                            .mapv_inplace(|x| x / norm);
                    }
                }
            }
        }
    }
    let start = Instant::now();
    for (node_iter, node_array) in local_max_vectors.axis_iter(Axis(0)).enumerate() {
        let pointing_space_vector_local =
            pointing_space_vector.slice(s![node_iter, .., .., .., ..]);
        for (_, target_vector_node) in node_array.axis_iter(Axis(0)).enumerate() {
            for x in 0..pointing_space_vector_local.shape()[0] {
                for y in 0..pointing_space_vector_local.shape()[1] {
                    for z in 0..pointing_space_vector_local.shape()[2] {
                        let specific_vector: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 1]>> =
                            pointing_space_vector_local.slice(s![x, y, z, ..]);
                        let destination = fusion_result.get_mut([x, y, z]).unwrap();
                        let dot_product = target_vector_node.dot(&specific_vector);
                        if dot_product.acos() * 180.0 / PI < 10.0 {
                            *destination += dot_product;
                        }
                    }
                }
            }
        }
    }
    println!("{} , passed time : {:?}", fusion_result, start.elapsed());
}

fn linspace(x0: f64, end: f64, count: u32) -> Vec<f64> {
    let dx = (end - x0) / f64::from(count - 1);
    (0..count)
        .into_iter()
        .map(|x| x0 + dx * f64::from(x))
        .collect_vec()
}

fn pure_for_style() {
    let nodes_locs = vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 4.0, 4.0, 0.0];
    let targets_locs = vec![-5.0, -5.0, 10.0, 5.0, 5.0, 5.0, 0.0, 5.0, 10.0];
    let mut local_max_vectors: Vec<f64> = vec![0.0; (nodes_locs.len() / 3) * targets_locs.len()];

    for node_index_iter in (0..nodes_locs.len()).step_by(3) {
        for target_index_iter in (0..targets_locs.len()).step_by(3) {
            let mut sum: f64 = 0.0;
            for dimension in 0..3 {
                local_max_vectors
                    [node_index_iter * targets_locs.len() / 3 + target_index_iter + dimension] =
                    targets_locs[target_index_iter + dimension]
                        - nodes_locs[node_index_iter + dimension];
                sum += local_max_vectors
                    [node_index_iter * targets_locs.len() / 3 + target_index_iter + dimension]
                    .powi(2);
            }
            sum = sum.sqrt();
            for dimension in 0..3 {
                local_max_vectors
                    [node_index_iter * targets_locs.len() / 3 + target_index_iter + dimension] /=
                    sum;
            }
        }
    }

    //make meshgrid
    let x_dimension = 256;
    let y_dimension = 256;
    let z_dimension = 128;

    //let points_count = x_dimension*y_dimension*z_dimension;
    let mut roi_points = Vec::<(f64, f64, f64)>::new();
    let x = linspace(-10.0, 10.0, x_dimension as u32);
    let y = x.clone();
    let z = linspace(0.0, 10.0, z_dimension as u32);

    for x_elem in x {
        for y_elem in &y {
            for z_elem in &z {
                roi_points.push((x_elem, *y_elem, *z_elem));
            }
        }
    }
    //size meshgrid * nodes_count
    let mut roi_vectors = Vec::<(f64, f64, f64)>::new();
    for roi_point in roi_points {
        for node_index_iter in (0..nodes_locs.len()).step_by(3) {
            let norm = ((roi_point.0 - nodes_locs[node_index_iter + 0]).powi(2)
                + (roi_point.1 - nodes_locs[node_index_iter + 1]).powi(2)
                + (roi_point.2 - nodes_locs[node_index_iter + 2]).powi(2))
            .sqrt();
            roi_vectors.push((
                (roi_point.0 - nodes_locs[node_index_iter + 0]) / norm,
                (roi_point.1 - nodes_locs[node_index_iter + 1]) / norm,
                (roi_point.2 - nodes_locs[node_index_iter + 2]) / norm,
            ));
        }
    }
    let mut fusion_result = vec![0.0; roi_vectors.len()];
    let start = Instant::now();

    for node in 0..nodes_locs.len() / 3 {
        for target in 0..targets_locs.len() / 3 {
            for roi_iter in (0..roi_vectors.len()).step_by(4) {
                //for (roi_iter , roi_vector) in roi_vectors.iter().enumerate(){
                let roi_vector = &roi_vectors[roi_iter..roi_iter + 4];
                let x_mul = f64x4::from_array([
                    roi_vector[0].0,
                    roi_vector[1].0,
                    roi_vector[2].0,
                    roi_vector[3].0,
                ]);
                let x_target_mul = f64x4::from_array(
                    [local_max_vectors[(node * targets_locs.len() / 3 + target) * 3 + 0]; 4],
                );
                let y_mul = f64x4::from_array([
                    roi_vector[0].1,
                    roi_vector[1].1,
                    roi_vector[2].1,
                    roi_vector[3].1,
                ]);
                let y_target_mul = f64x4::from_array(
                    [local_max_vectors[(node * targets_locs.len() / 3 + target) * 3 + 1]; 4],
                );
                let z_mul = f64x4::from_array([
                    roi_vector[0].2,
                    roi_vector[1].2,
                    roi_vector[2].2,
                    roi_vector[3].2,
                ]);
                let z_target_mul = f64x4::from_array(
                    [local_max_vectors[(node * targets_locs.len() / 3 + target) * 3 + 2]; 4],
                );
                let res_x = x_mul * x_target_mul;
                let res_y = y_mul * y_target_mul;
                let res_z = z_mul * z_target_mul;
                let res_final = (res_x + res_y + res_z).to_array();
                for i in 0..4 {
                    if res_final[i] * 180.0 / PI < 10.0 {
                        fusion_result[roi_iter + i] += res_final[i];
                    }
                }
                // let dot_product = roi_vector.0 * local_max_vectors[(node*targets_locs.len()/3 + target)*3 + 0] + roi_vector.1 * local_max_vectors[(node*targets_locs.len()/3 + target)*3 + 1] +
                // roi_vector.2 * local_max_vectors[(node*targets_locs.len()/3 + target)*3 + 2] ;
                // if dot_product.acos() * 180.0 / PI < 10.0 {
                // fusion_result[roi_iter] += dot_product;
                // }
            }
        }
    }
    println!("ELAPSED TIME : {:?} ", start.elapsed())
}

fn read_objects_location(file_name: &str) -> Vec<(f64, f64, f64)> {
    let mut target_1_loc = Vec::<(f64, f64, f64)>::new();
    let file = File::open(file_name).unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(Box::new(file));
    for result in rdr.records() {
        let record = result.unwrap();
        target_1_loc.push((
            record[0].parse().unwrap(),
            record[1].parse().unwrap(),
            record[2].parse().unwrap(),
        ));
    }
    target_1_loc
}

fn read_simulation_csv() -> (
    Vec<f64>,
    (
        Vec<(f64, f64, f64)>,
        Vec<(f64, f64, f64)>,
        Vec<(f64, f64, f64)>,
    ),
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
) {
    static NODES_COUNT: usize = 4;
    let mut nodes_locs = vec![0.0; NODES_COUNT * 3];
    let file = File::open(r".\simulation_aggregator\nodes_pos.csv").unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(Box::new(file));
    for (iter_result, result) in rdr.records().enumerate() {
        let record = result.unwrap();
        for i in 0..NODES_COUNT {
            nodes_locs[i * 3 + iter_result] = record[i].parse().unwrap();
        }
    }
    let obj_locs_1 = read_objects_location(r".\simulation_aggregator\pos_obj_1.csv");
    let obj_locs_2 = read_objects_location(r".\simulation_aggregator\pos_obj_2.csv");
    let obj_locs_3 = read_objects_location(r".\simulation_aggregator\pos_obj_3.csv");

    let mut phis_vector_target = Vec::<Vec<f64>>::new();
    let mut thetas_vector_target = Vec::<Vec<f64>>::new();
    let file = File::open(r".\simulation_aggregator\phis.csv").unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(Box::new(file));
    for result in rdr.records() {
        let record = result.unwrap();
        phis_vector_target.push(record.iter().map(|x| x.parse().unwrap()).collect());
    }
    let file = File::open(r".\simulation_aggregator\thetas.csv").unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(Box::new(file));
    for result in rdr.records() {
        let record = result.unwrap();
        thetas_vector_target.push(record.iter().map(|x| x.parse().unwrap()).collect());
    }
    (
        nodes_locs,
        (obj_locs_1, obj_locs_2, obj_locs_3),
        phis_vector_target,
        thetas_vector_target,
    )
}

fn read_string() -> String {
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .expect("can not read user input");
    input
}

fn acos_approx(x:&f64)->f64 {
    let mut negate=-1.0;
    if *x<0.0 {
        negate = 1.0
    }
    let mut x_local = x.clone();
    x_local = x_local.abs();
    let mut ret = -0.0187293;
    ret = ret * x;
    ret = ret + 0.0742610;
    ret = ret * x;
    ret = ret - 0.2121144;
    ret = ret * x;
    ret = ret + 1.5707288;
    ret = ret * (1.0-x).sqrt();
    ret = ret - 2.0 * negate * ret;
    negate * 3.14159265358979 + ret
  }

fn pure_for_style_with_simdata(
    nodes_locs: Vec<f64>,
    phis_vector_target: Vec<Vec<f64>>,
    thetas_vector_target: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let mut vector_target_relative_to_nodes = Vec::<Vec<(f64, f64, f64)>>::new();
    for (phi_vector_target, theta_vector_target) in
        phis_vector_target.iter().zip(thetas_vector_target.iter())
    {
        // random noise addition
        let mut rng = thread_rng();
        let normal = Normal::new(0.5, 0.5).unwrap();
        let random_vars = (0..phi_vector_target.len()).map(|_| (normal.sample(&mut rng) , normal.sample(&mut rng) ) ).collect_vec();
        
        vector_target_relative_to_nodes.push(
            phi_vector_target
                .iter()
                .zip(theta_vector_target.iter())
                .zip(random_vars.iter())
                .map(|x| (((x.0.0+x.1.0)*PI/180.0).cos() * ((x.0.1+x.1.1)*PI/180.0).sin(), ((x.0.0+x.1.0)*PI/180.0).sin() * ((x.0.1+x.1.1)*PI/180.0).sin(),((x.0.1+x.1.1)*PI/180.0).cos()))
                //.map(|x| ((x.0*PI/180.0).cos() * (x.1*PI/180.0).sin(), (x.0*PI/180.0).sin() * (x.1*PI/180.0).sin(),(x.1*PI/180.0).cos()))
                .collect_vec(),
        );
    }

    //make meshgrid
    let x_dimension = 20;
    let y_dimension = 20;
    let z_dimension = 10;

    //let points_count = x_dimension*y_dimension*z_dimension;
    let mut roi_points = Vec::<(f64, f64, f64)>::new();
    let x = linspace(-10000.0, 15000.0, x_dimension as u32);
    let y = linspace(-10000.0, 30000.0, y_dimension as u32);
    let z: Vec<f64> = linspace(0.0, 2000.0, z_dimension as u32);

    for x_elem in x {
        for y_elem in &y {
            for z_elem in &z {
                roi_points.push((x_elem, *y_elem, *z_elem));
            }
        }
    }
    //size meshgrid * nodes_count
    let mut roi_vectors = Vec::<(f64, f64, f64)>::new();
    for roi_point in &roi_points {
        for node_index_iter in (0..nodes_locs.len()).step_by(3) {
            let norm = ((roi_point.0 - nodes_locs[node_index_iter + 0]).powi(2)
                + (roi_point.1 - nodes_locs[node_index_iter + 1]).powi(2)
                + (roi_point.2 - nodes_locs[node_index_iter + 2]).powi(2))
            .sqrt();
            roi_vectors.push((
                (roi_point.0 - nodes_locs[node_index_iter + 0]) / norm,
                (roi_point.1 - nodes_locs[node_index_iter + 1]) / norm,
                (roi_point.2 - nodes_locs[node_index_iter + 2]) / norm,
            ));
        }
    }
    let mut fusion_result = vec![
        vec![0.0; roi_vectors.len() / (nodes_locs.len() / 3)];
        vector_target_relative_to_nodes[0].len()
    ];
    for time_frame in 0..vector_target_relative_to_nodes[0].len() {
        let targets_count = vector_target_relative_to_nodes.len() / (nodes_locs.len() / 3);
        for (roi_iter, roi_vector) in roi_vectors.chunks(4).enumerate() {
            for node in 0..nodes_locs.len() / 3 {
                for target in 0..targets_count {
                    let dot_product = roi_vector[node].0
                        * vector_target_relative_to_nodes[node * targets_count + target]
                            [time_frame]
                            .0
                        + roi_vector[node].1
                            * vector_target_relative_to_nodes[node * targets_count + target]
                                [time_frame]
                                .1
                        + roi_vector[node].2
                            * vector_target_relative_to_nodes[node * targets_count + target]
                                [time_frame]
                                .2;
                    if dot_product.acos() * 180.0 / PI < 10.0 {
                        fusion_result[time_frame][roi_iter] += dot_product;
                    }
                }
            }
        }
        println!("done and dusted! iteration:{}", time_frame);
    }
    fusion_result
}
fn main() {
    let (
        nodes_locs,
        (obj_locs_1, obj_locs_2, obj_locs_3),
        phis_vector_target,
        thetas_vector_target,
    ) = read_simulation_csv();
    let result = pure_for_style_with_simdata(nodes_locs, phis_vector_target, thetas_vector_target);
    let mut wtr = Writer::from_path("fusion_result.csv").unwrap();
    for mut result_item in result {
        wtr.write_record(result_item.iter_mut().map(|x| x.to_string()).collect_vec())
            .unwrap();
    }
    wtr.flush().unwrap();
    //pure_for_style();
    //ndarray_style();
}
