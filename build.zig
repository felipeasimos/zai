const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/zai.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_mod.addImport("zai", lib_mod);
    exe_mod.addAnonymousImport("mnist-mini-csv", .{ .root_source_file = b.path("MNIST.csv") });

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "zai",
        .root_module = lib_mod,
    });

    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "zai",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const contract = b.dependency("contract", .{});
    const tensor = b.dependency("tensor", .{});
    const zcsv = b.dependency("zcsv", .{});
    const zglfw = b.dependency("zglfw", .{});
    const zgpu = b.dependency("zgpu", .{});
    const zgui = b.dependency("zgui", .{
        .shared = false,
        .backend = .glfw_wgpu,
        .with_implot = true,
    });

    lib.root_module.addImport("tensor", tensor.module("tensor"));

    exe.root_module.addImport("zcsv", zcsv.module("zcsv"));
    exe.root_module.addImport("zglfw", zglfw.module("root"));
    exe.root_module.addImport("zgpu", zgpu.module("root"));
    exe.root_module.addImport("zgui", zgui.module("root"));
    exe.root_module.addImport("tensor", tensor.module("tensor"));
    exe.root_module.addImport("contract", contract.module("contract"));

    @import("zgpu").addLibraryPathsTo(exe);

    if (target.result.os.tag != .emscripten) {
        exe.linkLibrary(zglfw.artifact("glfw"));
        exe.linkLibrary(zgpu.artifact("zdawn"));
        exe.linkLibrary(zgui.artifact("imgui"));
    }

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
