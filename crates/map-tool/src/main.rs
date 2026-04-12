use iced::{
    widget::{canvas, column, container, stack, text},
    Color, ContentFit, Element, Length, Point, Rectangle, Renderer, Size, Theme,
};
use std::env;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Message {
    /// Carries (x_pct, y_pct) packed into a Point.
    CursorMoved(Point),
    /// Left-click: print coordinates to stdout. Carries (x_pct, y_pct).
    Clicked(Point),
}

struct App {
    image_handle: iced::widget::image::Handle,
    img_w: f32,
    img_h: f32,
    /// Last cursor percentage position (x in 0..1, y in 0..1). None = outside image.
    cursor_pct: Option<(f32, f32)>,
}

impl App {
    fn new() -> (Self, iced::Task<Message>) {
        let mut args = env::args().skip(1);
        let image_path = args
            .next()
            .expect("Usage: map-tool <image_path> [width] [height]");
        let img_w: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1536.0);
        let img_h: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2048.0);

        let image_handle = iced::widget::image::Handle::from_path(&image_path);

        (
            Self {
                image_handle,
                img_w,
                img_h,
                cursor_pct: None,
            },
            iced::Task::none(),
        )
    }

    fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            Message::CursorMoved(pct) => {
                self.cursor_pct = Some((pct.x, pct.y));
            }
            Message::Clicked(pct) => {
                println!("x: {:.4}  y: {:.4}", pct.x, pct.y);
            }
        }
        iced::Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let coord_str = match self.cursor_pct {
            Some((x, y)) => format!("x: {x:.4}   y: {y:.4}   (left-click to print)"),
            None => "Move cursor over the image".to_string(),
        };

        let overlay = CoordOverlay {
            img_w: self.img_w,
            img_h: self.img_h,
        };

        let map = stack![
            iced::widget::image(self.image_handle.clone())
                .width(Length::Fill)
                .height(Length::Fill)
                .content_fit(ContentFit::Contain),
            canvas(overlay).width(Length::Fill).height(Length::Fill),
        ];

        let status_bar = container(text(coord_str).size(18).color(Color::WHITE))
            .style(|_theme: &Theme| container::Style {
                background: Some(Color::from_rgb(0.1, 0.1, 0.1).into()),
                ..Default::default()
            })
            .width(Length::Fill)
            .padding([6, 12]);

        column![map, status_bar].into()
    }
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

/// Compute the ContentFit::Contain rendered dimensions and offsets within a canvas.
/// Returns (disp_w, disp_h, offset_x, offset_y).
fn contain_rect(canvas_w: f32, canvas_h: f32, img_w: f32, img_h: f32) -> (f32, f32, f32, f32) {
    let img_ratio = img_w / img_h;
    let canvas_ratio = canvas_w / canvas_h;
    let (disp_w, disp_h) = if canvas_ratio < img_ratio {
        let s = canvas_w / img_w;
        (canvas_w, img_h * s)
    } else {
        let s = canvas_h / img_h;
        (img_w * s, canvas_h)
    };
    let offset_x = (canvas_w - disp_w) / 2.0;
    let offset_y = (canvas_h - disp_h) / 2.0;
    (disp_w, disp_h, offset_x, offset_y)
}

/// Convert a canvas-local point to image-relative percentages (0.0–1.0).
/// Returns None when the point is in the letterbox area outside the image.
fn to_pct(local: Point, disp_w: f32, disp_h: f32, off_x: f32, off_y: f32) -> Option<(f32, f32)> {
    let x = (local.x - off_x) / disp_w;
    let y = (local.y - off_y) / disp_h;
    ((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)).then_some((x, y))
}

// ---------------------------------------------------------------------------
// Canvas overlay — transparent, just handles mouse events
// ---------------------------------------------------------------------------

struct CoordOverlay {
    img_w: f32,
    img_h: f32,
}

impl canvas::Program<Message> for CoordOverlay {
    type State = ();

    fn update(
        &self,
        _state: &mut (),
        event: canvas::Event,
        bounds: Rectangle,
        cursor: iced::mouse::Cursor,
    ) -> (canvas::event::Status, Option<Message>) {
        // Convert absolute cursor position to percentage message, if inside bounds/image.
        let pct_msg = |make: fn(Point) -> Message| -> Option<Message> {
            cursor.position_in(bounds).and_then(|local| {
                let (disp_w, disp_h, off_x, off_y) =
                    contain_rect(bounds.width, bounds.height, self.img_w, self.img_h);
                to_pct(local, disp_w, disp_h, off_x, off_y).map(|(x, y)| make(Point::new(x, y)))
            })
        };

        match event {
            canvas::Event::Mouse(iced::mouse::Event::CursorMoved { .. }) => (
                canvas::event::Status::Ignored,
                pct_msg(Message::CursorMoved),
            ),
            canvas::Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
                (canvas::event::Status::Ignored, pct_msg(Message::Clicked))
            }
            _ => (canvas::event::Status::Ignored, None),
        }
    }

    fn draw(
        &self,
        _state: &(),
        _renderer: &Renderer,
        _theme: &Theme,
        _bounds: Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        // The image widget underneath handles rendering; this canvas is transparent.
        vec![]
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main() -> iced::Result {
    iced::application("Map Tool", App::update, App::view)
        .window(iced::window::Settings {
            size: Size::new(700.0, 900.0),
            ..Default::default()
        })
        .run_with(App::new)
}
