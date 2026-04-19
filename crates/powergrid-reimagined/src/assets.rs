use bevy::{
    prelude::*,
    render::{render_asset::RenderAssetUsages, render_resource::TextureFormat},
};
use bevy_egui::EguiUserTextures;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Bevy resources
// ---------------------------------------------------------------------------

/// Keeps the map image handle alive so the GPU texture is not dropped.
#[derive(Resource)]
pub struct MapImageHandle(#[allow(dead_code)] pub Handle<Image>);

/// Keeps card image handles alive so GPU textures are not dropped.
#[derive(Resource)]
pub struct CardHandles(#[allow(dead_code)] pub HashMap<u8, Handle<Image>>);

#[derive(Resource)]
pub struct EguiMapTexture(pub egui::TextureId);

#[derive(Resource)]
pub struct EguiCardTextures(pub HashMap<u8, egui::TextureId>);

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

pub fn setup_assets(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut egui_textures: ResMut<EguiUserTextures>,
) {
    // ---- Map ----
    let map_bytes: &[u8] = include_bytes!("../assets/maps/germany.png");
    let map_image = load_png(map_bytes);
    let map_handle = images.add(map_image);
    let map_egui_id = egui_textures.add_image(map_handle.clone());
    commands.insert_resource(MapImageHandle(map_handle));
    commands.insert_resource(EguiMapTexture(map_egui_id));

    // ---- Cards ----
    macro_rules! card {
        ($n:expr, $path:expr) => {
            ($n, include_bytes!($path) as &[u8])
        };
    }

    let card_data: &[(u8, &[u8])] = &[
        card!(3, "../assets/cards/card_03.png"),
        card!(4, "../assets/cards/card_04.png"),
        card!(5, "../assets/cards/card_05.png"),
        card!(6, "../assets/cards/card_06.png"),
        card!(7, "../assets/cards/card_07.png"),
        card!(8, "../assets/cards/card_08.png"),
        card!(9, "../assets/cards/card_09.png"),
        card!(10, "../assets/cards/card_10.png"),
        card!(11, "../assets/cards/card_11.png"),
        card!(12, "../assets/cards/card_12.png"),
        card!(13, "../assets/cards/card_13.png"),
        card!(14, "../assets/cards/card_14.png"),
        card!(15, "../assets/cards/card_15.png"),
        card!(16, "../assets/cards/card_16.png"),
        card!(17, "../assets/cards/card_17.png"),
        card!(18, "../assets/cards/card_18.png"),
        card!(19, "../assets/cards/card_19.png"),
        card!(20, "../assets/cards/card_20.png"),
        card!(21, "../assets/cards/card_21.png"),
        card!(22, "../assets/cards/card_22.png"),
        card!(23, "../assets/cards/card_23.png"),
        card!(24, "../assets/cards/card_24.png"),
        card!(25, "../assets/cards/card_25.png"),
        card!(26, "../assets/cards/card_26.png"),
        card!(27, "../assets/cards/card_27.png"),
        card!(28, "../assets/cards/card_28.png"),
        card!(29, "../assets/cards/card_29.png"),
        card!(30, "../assets/cards/card_30.png"),
        card!(31, "../assets/cards/card_31.png"),
        card!(32, "../assets/cards/card_32.png"),
        card!(33, "../assets/cards/card_33.png"),
        card!(34, "../assets/cards/card_34.png"),
        card!(35, "../assets/cards/card_35.png"),
        card!(36, "../assets/cards/card_36.png"),
        card!(37, "../assets/cards/card_37.png"),
        card!(38, "../assets/cards/card_38.png"),
        card!(39, "../assets/cards/card_39.png"),
        card!(40, "../assets/cards/card_40.png"),
        card!(42, "../assets/cards/card_42.png"),
        card!(44, "../assets/cards/card_44.png"),
        card!(46, "../assets/cards/card_46.png"),
        card!(50, "../assets/cards/card_50.png"),
        card!(0, "../assets/cards/card_step3.png"),
    ];

    let mut card_handles = HashMap::new();
    let mut card_egui_ids = HashMap::new();

    for &(num, bytes) in card_data {
        let image = load_png(bytes);
        let handle = images.add(image);
        let egui_id = egui_textures.add_image(handle.clone());
        card_handles.insert(num, handle);
        card_egui_ids.insert(num, egui_id);
    }

    commands.insert_resource(CardHandles(card_handles));
    commands.insert_resource(EguiCardTextures(card_egui_ids));
}

// ---------------------------------------------------------------------------
// Helper: decode PNG bytes into a Bevy Image via the `image` crate
// ---------------------------------------------------------------------------

fn load_png(bytes: &[u8]) -> Image {
    use bevy::render::render_resource::{Extent3d, TextureDimension};
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .expect("failed to decode embedded PNG")
        .into_rgba8();
    let width = img.width();
    let height = img.height();
    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        img.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}
