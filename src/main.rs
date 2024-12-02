use wgpu_nnd::{ cli, WgslArgs, WgslSlices };
use std::mem::size_of_val;
use wgpu::util::DeviceExt;

async fn run() {
    let (info, mut buffers_input) = cli();
    let knn = buffers_input.knn.as_mut_slice();
    let data = buffers_input.data.as_mut_slice();
    let distances = buffers_input.distances.as_mut_slice();
    let buffers = WgslSlices {
        scratch: buffers_input.scratch.as_mut_slice(),
        avl: buffers_input.avl.as_mut_slice(),
        meta: buffers_input.meta.as_mut_slice(),
    };
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .unwrap();

    let shader = device.create_shader_module(
        wgpu::include_wgsl!("shader.wgsl"));

    let storage_buffer_data = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&data[..]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let storage_buffer_distances = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&distances[..]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let storage_buffer_knn = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&knn[..]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_of_val(knn) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    debug_assert!(
        size_of_val(buffers.scratch) as u32 / 4 == info.scratch_info.offset +
        info.scratch_info.row_strides * info.scratch_info.rows);
    debug_assert!(
        size_of_val(buffers.avl) as u32 / 4 == info.avl_info.offset +
        info.avl_info.row_strides * info.avl_info.rows);
    debug_assert!(
        size_of_val(buffers.meta) as u32 / 4 == info.meta_info.offset +
        info.meta_info.row_strides * info.meta_info.rows);
    let storage_buffer_scratch = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                &buffers.scratch[..], &buffers.avl, &buffers.meta].concat()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    #[cfg(debug_assertions)]
    let debug_distances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_of_val(distances) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    #[cfg(debug_assertions)]
    let scratch_size = size_of_val(buffers.scratch) + size_of_val(buffers.avl) +
        size_of_val(buffers.meta);
    #[cfg(debug_assertions)]
    let debug_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: scratch_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer_data.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage_buffer_distances.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: storage_buffer_knn.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: storage_buffer_scratch.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let avl_offset = info.scratch_info.offset +
        info.scratch_info.row_strides * info.scratch_info.rows;
    let meta_offset = avl_offset + info.avl_info.offset +
        info.avl_info.row_strides * info.avl_info.rows;
    let pipeline_options = wgpu::PipelineCompilationOptions {
        constants: &[
            ("k".to_owned(), info.knn_info.cols.into()),
            ("candidates".to_owned(), info.candidates.into()),
            ("ndim".to_owned(), info.data_info.cols.into()),
            ("points".to_owned(), info.data_info.rows.into()),

            ("data_offset".to_owned(), info.data_info.offset.into()),
            ("data_row_strides".to_owned(), info.data_info.row_strides.into()),
            ("data_col_strides".to_owned(), info.data_info.col_strides.into()),

            ("distances_offset".to_owned(), info.distances_info.offset.into()),
            ("distances_row_strides".to_owned(),
                info.distances_info.row_strides.into()),
            ("distances_col_strides".to_owned(),
                info.distances_info.col_strides.into()),

            ("knn_offset".to_owned(), info.knn_info.offset.into()),
            ("knn_row_strides".to_owned(), info.knn_info.row_strides.into()),
            ("knn_col_strides".to_owned(), info.knn_info.col_strides.into()),

            ("scratch_offset".to_owned(), info.scratch_info.offset.into()),
            ("scratch_row_strides".to_owned(),
                info.scratch_info.row_strides.into()),
            ("scratch_col_strides".to_owned(),
                info.scratch_info.col_strides.into()),

            ("avl_offset".to_owned(), avl_offset.into()),
            ("avl_row_strides".to_owned(), info.avl_info.row_strides.into()),
            ("avl_col_strides".to_owned(), info.avl_info.col_strides.into()),
            ("avl_vox_strides".to_owned(), info.avl_info.vox_strides.into()),

            ("meta_offset".to_owned(), meta_offset.into()),
            ("meta_row_strides".to_owned(), info.meta_info.row_strides.into()),
            ("meta_col_strides".to_owned(), info.meta_info.col_strides.into()),
        ].into(), ..Default::default()};
    let pipeline = device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: pipeline_options,
            cache: None,
        });

    //----------------------------------------------------------

    let mut command_encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(Some(command_encoder.finish()));

    //----------------------------------------------------------

    get_data(
        &mut knn[..],
        &storage_buffer_knn,
        &output_staging_buffer,
        &device,
        &queue,
    )
    .await;
    #[cfg(debug_assertions)]
    let mut distances = unsafe {
        Box::<[f32]>::new_uninit_slice(size_of_val(distances) / 4).assume_init()
    };
    #[cfg(debug_assertions)]
    get_data(
        &mut distances[..],
        &storage_buffer_distances,
        &debug_distances_buffer,
        &device,
        &queue,
    )
    .await;
    #[cfg(debug_assertions)]
    let mut debug = unsafe {
        Box::<[i32]>::new_uninit_slice(scratch_size / 4).assume_init()
    };
    #[cfg(debug_assertions)]
    get_data(
        &mut debug[..],
        &storage_buffer_scratch,
        &debug_staging_buffer,
        &device,
        &queue,
    )
    .await;

    log::info!("Output: {:?}", knn);
    #[cfg(debug_assertions)]
    visualize(&info, &knn, &distances, &debug);
}

async fn get_data<T: bytemuck::Pod>(
    output: &mut [T],
    storage_buffer: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut command_encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: None });
    command_encoder.copy_buffer_to_buffer(
        storage_buffer, 0, staging_buffer, 0, size_of_val(output) as u64);
    queue.submit(Some(command_encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(
        wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    receiver.recv_async().await.unwrap().unwrap();
    output.copy_from_slice(
        bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
    staging_buffer.unmap();
}

fn visualize(info: &WgslArgs, knn: &[i32], distances: &[f32], debug: &[i32]) {
    log::info!("Distances: {:?}", distances);
    let avl_offset = (info.scratch_info.offset +
        info.scratch_info.row_strides * info.scratch_info.rows) as usize;
    let meta_offset = (avl_offset as u32 + info.avl_info.offset +
        info.avl_info.row_strides * info.avl_info.rows) as usize;
    let candidates = &debug[info.scratch_info.offset as usize..avl_offset];
    let avl = &debug[avl_offset + (info.avl_info.offset as usize)..meta_offset];
    let meta = &debug[meta_offset + (info.meta_info.offset as usize)..];
    log::info!("Candidates: {:?}", candidates);
    log::info!("Meta: {:?}", meta);
    for i in 0..info.knn_info.rows {
        let row_knn = &knn[
                (i * info.knn_info.row_strides) as usize..
                (i * info.knn_info.row_strides +
                    info.distances_info.col_strides *
                    info.knn_info.cols) as usize
            ];
        let row_distances = &distances[
                (i * info.distances_info.row_strides) as usize..
                (i * info.distances_info.row_strides +
                    info.distances_info.col_strides *
                    info.distances_info.cols) as usize
            ];
        let row_avl = &avl[
                (i * info.avl_info.row_strides) as usize..
                (i * info.avl_info.row_strides +
                    info.avl_info.col_strides *
                    info.avl_info.cols) as usize
            ];
        let tree = walk(
            row_knn, row_distances, row_avl,
            info.data_info.rows as i32,
            info.avl_info.col_strides as usize,
            meta[(info.meta_info.row_strides * i) as usize],
            String::from(""),
            String::from("")
        );
        log::info!("AVL: {:?}", row_avl);
        println!("{}", &tree[0..std::cmp::max(tree.len(), 1) - 1]);
    }
}

fn walk(
    knn: &[i32],
    distances: &[f32],
    avl: &[i32],
    points: i32,
    strides: usize,
    node: i32,
    prefix: String,
    postfix: String
) -> String {
    if node == -1 { return "".to_owned(); }
    let mut line = format!(
        "{}{}", &prefix[0..std::cmp::max(prefix.len(), 3) - 3], postfix);
    let h = avl[node as usize * strides + 0];
    let l = avl[node as usize * strides + 1];
    let r = avl[node as usize * strides + 2];
    let u = avl[node as usize * strides + 3];
    if l == -1 && r == -1 {
        line.push_str(&format!(
            "\u{2500}({}^{}h{}) {} {}\n",
            node, u, h, knn[node as usize] % points, distances[node as usize]));
    } else {
        line.push_str(&format!(
            "\u{252C}({}^{}h{}) {} {}\n",
            node, u, h, knn[node as usize] % points, distances[node as usize]));
        line.push_str(&walk(knn, distances, avl, points, strides, l, format!(
            "{}\u{2502}", prefix), "\u{251C}".to_owned()));
        line.push_str(&walk(knn, distances, avl, points, strides, r, format!(
            "{}\u{2007}", prefix), "\u{2514}".to_owned()));
    }
    line
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect(
            "could not initialize logger");

        crate::utils::add_web_nothing_to_see_msg();

        wasm_bindgen_futures::spawn_local(run());
    }
}
