const std = @import("std");
const Tensor = @import("tensor").Tensor;

pub fn TrainingParams(comptime dtype: type) type {
    return struct {
        learning_rate: dtype,
        random: std.Random,
    };
}

pub const FullyConnectedLayerOptions = struct {
    input_size: usize,
    output_size: usize,
    batch_size: usize,
    dtype: type,
    activation: struct {
        f: fn (anytype) void,
        prime: fn (anytype) void,
    },
    next_layer_type: ?type = null,
};

fn getComptimeFieldValue(comptime T: type, comptime field_name: []const u8) ?@FieldType(T, field_name) {
    const type_info = @typeInfo(T);
    inline for (type_info.@"struct".fields) |field| {
        if (std.mem.eql(u8, field.name, field_name)) {
            if (field.default_value_ptr) |default_ptr| {
                return @as(*const field.type, @ptrCast(@alignCast(default_ptr))).*;
            }
        }
    }
    return null;
}

pub fn FullyConnectedLayer(comptime options: FullyConnectedLayerOptions) type {
    const dtype = options.dtype;
    const weights_shape: [2]usize = .{ options.input_size, options.output_size };
    // const bias_shape = .{options.output_size};
    const input_shape = .{ options.batch_size, options.input_size };
    const output_shape: [2]usize = .{ options.batch_size, options.output_size };
    const is_final_layer = options.next_layer_type == null;

    const _final_output_shape = final_output_shape: {
        if (options.next_layer_type) |NextLayerType| {
            break :final_output_shape getComptimeFieldValue(NextLayerType, "final_output_shape").?;
        }
        break :final_output_shape output_shape;
    };

    const Data = data: {
        if (options.next_layer_type) |NextLayerType| {
            break :data struct {
                weights: Tensor(dtype, weights_shape) = undefined,
                input: Tensor(dtype, input_shape) = undefined,
                weight_gradients: Tensor(dtype, weights_shape) = undefined,

                preactivation: Tensor(dtype, output_shape) = undefined,
                sigma: Tensor(dtype, output_shape) = undefined,

                next_layer: NextLayerType,
            };
        }
        break :data struct {
            weights: Tensor(dtype, weights_shape) = undefined,
            input: Tensor(dtype, input_shape) = undefined,
            weight_gradients: Tensor(dtype, weights_shape) = undefined,

            preactivation: Tensor(dtype, output_shape) = undefined,
            sigma: Tensor(dtype, output_shape) = undefined,
        };
    };

    return struct {
        data: Data,
        comptime final_output_shape: @TypeOf(_final_output_shape) = _final_output_shape,
        comptime final_output_type: type = Tensor(dtype, _final_output_shape),

        pub fn init() @This() {
            if (comptime !is_final_layer) {
                return .{
                    .data = .{
                        .next_layer = @FieldType(Data, "next_layer").init(),
                    },
                };
            }
            return .{
                .data = .{},
            };
        }

        pub fn random(rand: std.Random) @This() {
            if (comptime !is_final_layer) {
                var new: @This() = .{
                    .data = .{
                        .next_layer = @FieldType(Data, "next_layer").random(rand),
                    },
                };
                new.data.weight_gradients.randomize(rand);
                new.data.weights.randomize(rand);
                new.data.input.randomize(rand);
                new.data.preactivation.randomize(rand);
                new.data.sigma.randomize(rand);
                return new;
            }
            var new: @This() = .{
                .data = .{},
            };
            new.data.weight_gradients.randomize(rand);
            new.data.weights.randomize(rand);
            new.data.input.randomize(rand);
            new.data.preactivation.randomize(rand);
            new.data.sigma.randomize(rand);
            return new;
        }

        // f(X * W + B)
        pub fn forward(self: *@This(), input: anytype, result: *self.final_output_type) void {
            self.data.input.copy(input);
            var output = input.matmulNew(&self.data.weights); // X * W
            self.data.preactivation.copy(&output); // save Z
            // output = A = f(Z)
            for (0..options.batch_size) |row| {
                var mut = output.mut(.{row});
                options.activation.f(&mut);
            }
            if (comptime !is_final_layer) {
                self.data.next_layer.forward(&output, result);
            }
        }

        pub fn layerBackprop(self: *@This(), guess: anytype, correct: anytype) dtype {
            var loss: dtype = 0;
            // sigma[i] = dL/dZ[i+1] = (sigma[i+1] @ W[i+1]) * dA[i+1]/dZ[i+1]
            // if i == o: sigma = -(y-ŷ) * dA[i+1]/dZ[i+1]
            // dL/W[i] = A[i].T @ sigma[i]
            if (comptime is_final_layer) {
                // calculate sigma
                guess.wise(correct, &self.data.sigma, (struct {
                    pub fn func(a: dtype, b: dtype) dtype {
                        return a - b;
                    }
                }).func);
                var tmp: @TypeOf(self.data.preactivation) = undefined;
                for (0..options.batch_size) |row| {
                    var mut = tmp.mut(.{row});
                    mut.copy(self.data.preactivation.view(.{row}));
                    options.activation.f(&mut);
                }
                self.data.sigma.wise(&tmp, &self.data.sigma, (struct {
                    pub fn func(loss_gradient: dtype, z: dtype) dtype {
                        return loss_gradient * z;
                    }
                }).func);

                // calculate loss
                var sum: dtype = 0;
                for (correct.data, guess.data) |d, g| {
                    const diff = d - g;
                    sum += diff * diff;
                }
                loss = sum / 2;
            } else {
                // calculate sigma
                self.data.next_layer.data.sigma.matmul(&self.data.next_layer.data.weights.transpose(.{}), &self.data.sigma);
                var tmp: @TypeOf(self.data.preactivation) = undefined;
                for (0..options.batch_size) |row| {
                    var mut = tmp.mut(.{row});
                    mut.copy(self.data.preactivation.view(.{row}));
                    options.activation.f(&mut);
                }
                self.data.sigma.wise(&tmp, &self.data.sigma, (struct {
                    pub fn func(z: dtype, sigma: dtype) dtype {
                        return sigma * z;
                    }
                }).func);
                loss = self.data.next_layer.layerBackprop(guess, correct);
            }
            // update gradients by multiplying a^t and matrix multiplying by sigma
            self.data.input
                .transpose(.{})
                .matmul(&self.data.sigma, &self.data.weight_gradients);

            return loss;
        }

        // update bias: sum of current sigma
        // update weights: activations transposed and current sigma
        // sigma: activation gradient, next layer sigma and next layer weight transposed
        pub fn backprop(self: *@This(), input: anytype, correct: anytype) dtype {
            var guess: Tensor(dtype, self.final_output_shape) = undefined;
            self.forward(input, &guess);
            return self.layerBackprop(&guess, correct);
        }

        pub fn layerTrain(self: *@This(), params: TrainingParams(dtype)) void {
            self.data.weight_gradients.wise(params.learning_rate, &self.data.weight_gradients, (struct {
                pub fn func(derivative: dtype, learning_rate: dtype) dtype {
                    return learning_rate * derivative;
                }
            }).func);
            self.data.weights.wise(&self.data.weight_gradients, &self.data.weights, (struct {
                pub fn func(weight: dtype, derivate: dtype) dtype {
                    return weight - derivate;
                }
            }).func);
            if (comptime !is_final_layer) {
                self.data.next_layer.layerTrain(params);
            }
        }

        pub fn train(self: *@This(), input: anytype, correct: anytype, params: TrainingParams(dtype)) dtype {
            const num_samples = options.batch_size;

            var batch_x: Tensor(dtype, input_shape) = undefined;
            var batch_y: Tensor(dtype, self.final_output_shape) = undefined;
            for (0..num_samples) |i| {
                const idx = params.random.uintLessThan(usize, num_samples);
                var data_sample = input.clone(.{idx});
                var target_sample = correct.clone(.{idx});
                var batch_x_row = batch_x.mut(.{i});
                batch_x_row.copy(&data_sample);
                var batch_y_row = batch_y.mut(.{i});
                batch_y_row.copy(&target_sample);
            }
            const loss = self.backprop(&batch_x, &batch_y);
            // const loss = self.backprop(input, correct);
            self.layerTrain(params);
            return loss;
        }
    };
}
