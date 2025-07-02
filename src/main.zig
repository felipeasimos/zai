const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zai = @import("zai");
const zcsv = @import("zcsv");
const Tensor = @import("tensor").Tensor;

const mnist = @embedFile("mnist-mini-csv");

const dtype = f32;

const window_title = "zig-gamedev: minimal zgpu zgui";

fn sigmoidPure(d: dtype) dtype {
    const n = std.math.clamp(d, -500, 500);
    return 1 / (1 + std.math.exp(-n));
}

fn sigmoid(a: anytype) void {
    a.apply(sigmoidPure);
}
fn sigmoidPrime(a: anytype) void {
    a.apply((struct {
        pub fn func(d: dtype) dtype {
            const n = std.math.clamp(d, -500, 500);
            return sigmoid(n) * (1 - sigmoid(n));
        }
    }).func);
}
fn relu(a: anytype) void {
    a.apply((struct {
        pub fn func(d: dtype) dtype {
            return @max(d, 0);
        }
    }).func);
}
fn reluPrime(a: anytype) void {
    a.apply((struct {
        pub fn func(d: dtype) dtype {
            return if (d > 0) 1 else 0;
        }
    }).func);
}

fn train() !void {
    var parser = zcsv.zero_allocs.slice.init(mnist, .{});
    const seed = 1337;
    var prng = std.Random.DefaultPrng.init(blk: {
        std.crypto.random.bytes(std.mem.asBytes(&seed));
        break :blk seed;
    });
    var fully_connected_layer = zai.FullyConnectedLayer(zai.FullyConnectedLayerOptions{
        .dtype = dtype,
        .input_size = 64,
        .output_size = 32,
        .batch_size = 32,
        .activation = .{
            .f = relu,
            .prime = reluPrime,
        },
        .next_layer_type = zai.FullyConnectedLayer(zai.FullyConnectedLayerOptions{
            .dtype = dtype,
            .input_size = 32,
            .output_size = 10,
            .batch_size = 32,
            .activation = .{
                .f = sigmoid,
                .prime = sigmoidPrime,
            },
        }),
    }).random(prng.random());

    var x = Tensor(dtype, .{ 1797, 64 }){ .data = undefined };
    var y = Tensor(dtype, .{ 1797, 10 }){ .data = undefined };
    _ = parser.next(); // ignore header
    var i: usize = 0;
    while (parser.next()) |row| : (i += 1) {
        var j: usize = 0;
        var iter = row.iter();
        while (iter.next()) |field| : (j += 1) {
            const value: dtype = try std.fmt.parseFloat(dtype, field.raw());
            if (j == 64) {
                const value_idx = @as(usize, @intFromFloat(value));
                y.scalar(.{ i, value_idx }).* = 1;
            } else {
                x.scalar(.{ i, j }).* = value;
            }
        }
    }
    var loss: f64 = 1;
    i = 0;
    while (loss > 1e-4 and i < 3) : (i += 1) {
        @import("std").debug.print("i: {}, loss: {}\n", .{ i, loss });
        loss = fully_connected_layer.train(&x, &y, .{
            .learning_rate = 1e-4,
            .random = prng.random(),
        });
    }
}
pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    // Change current working directory to where the executable is located.
    {
        var buffer: [1024]u8 = undefined;
        const path = std.fs.selfExeDirPath(buffer[0..]) catch ".";
        std.posix.chdir(path) catch {};
    }

    zglfw.windowHint(.client_api, .no_api);

    const window = try zglfw.Window.create(800, 500, window_title, null);
    defer window.destroy();
    window.setSizeLimits(400, 400, -1, -1);

    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const gctx = try zgpu.GraphicsContext.create(
        gpa,
        .{
            .window = window,
            .fn_getTime = @ptrCast(&zglfw.getTime),
            .fn_getFramebufferSize = @ptrCast(&zglfw.Window.getFramebufferSize),
            .fn_getWin32Window = @ptrCast(&zglfw.getWin32Window),
            .fn_getX11Display = @ptrCast(&zglfw.getX11Display),
            .fn_getX11Window = @ptrCast(&zglfw.getX11Window),
            .fn_getWaylandDisplay = @ptrCast(&zglfw.getWaylandDisplay),
            .fn_getWaylandSurface = @ptrCast(&zglfw.getWaylandWindow),
            .fn_getCocoaWindow = @ptrCast(&zglfw.getCocoaWindow),
        },
        .{},
    );
    defer gctx.destroy(gpa);

    const scale_factor = scale_factor: {
        const scale = window.getContentScale();
        break :scale_factor @max(scale[0], scale[1]);
    };

    zgui.init(gpa);
    defer zgui.deinit();

    _ = zgui.io.addFontFromFile(
        "../../Roboto-Medium.ttf",
        std.math.floor(16.0 * scale_factor),
    );

    zgui.backend.init(
        window,
        gctx.device,
        @intFromEnum(zgpu.GraphicsContext.swapchain_format),
        @intFromEnum(wgpu.TextureFormat.undef),
    );
    defer zgui.backend.deinit();

    zgui.getStyle().scaleAllSizes(scale_factor);

    try train();

    while (!window.shouldClose() and window.getKey(.escape) != .press) {
        zglfw.pollEvents();

        zgui.backend.newFrame(
            gctx.swapchain_descriptor.width,
            gctx.swapchain_descriptor.height,
        );

        // Set the starting window position and size to custom values
        zgui.setNextWindowPos(.{ .x = 20.0, .y = 20.0, .cond = .first_use_ever });
        zgui.setNextWindowSize(.{ .w = -1.0, .h = -1.0, .cond = .first_use_ever });

        if (zgui.begin("My window", .{})) {
            if (zgui.button("Press me!", .{ .w = 200.0 })) {
                std.debug.print("Button pressed\n", .{});
            }
        }
        zgui.end();

        const swapchain_texv = gctx.swapchain.getCurrentTextureView();
        defer swapchain_texv.release();

        const commands = commands: {
            const encoder = gctx.device.createCommandEncoder(null);
            defer encoder.release();

            // GUI pass
            {
                const pass = zgpu.beginRenderPassSimple(encoder, .load, swapchain_texv, null, null, null);
                defer zgpu.endReleasePass(pass);
                zgui.backend.draw(pass);
            }

            break :commands encoder.finish(null);
        };
        defer commands.release();

        gctx.submit(&.{commands});
        _ = gctx.present();
    }
}
