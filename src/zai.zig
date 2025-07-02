const std = @import("std");
const Tensor = @import("tensor").Tensor;
const contract = @import("contract");

pub fn TrainingParams(comptime dtype: type) type {
    return struct {
        learning_rate: dtype,
        random: std.Random,
    };
}

pub fn ActivationFunctionSpec(comptime dtype: type) type {
    return struct {
        f: fn (dtype) dtype,
        prime: fn (dtype) dtype,
    };
}

pub fn FullyConnectedLayerOptions(comptime dtype: type) type {
    return struct {
        input_size: usize,
        output_size: usize,
        batch_size: usize,
        activation: ActivationFunctionSpec(dtype),
        next_layer_type: ?type = null,
    };
}

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

pub fn FullyConnectedLayer(comptime dtype: type, comptime options: FullyConnectedLayerOptions(dtype)) type {
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
                return new;
            }
            var new: @This() = .{
                .data = .{},
            };
            new.data.weight_gradients.randomize(rand);
            return new;
        }

        // f(X * W + B)
        pub fn forward(self: *@This(), input: anytype, result: *self.final_output_type) void {
            self.data.input.copy(input);
            var output = input.matmulNew(&self.data.weights); // X * W
            self.data.preactivation.copy(&output); // save Z
            output.apply(options.activation.f); // output = A = f(Z)

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
                self.data.sigma.wise(&self.data.preactivation, &self.data.sigma, (struct {
                    pub fn func(loss_gradient: dtype, z: dtype) dtype {
                        return loss_gradient * options.activation.prime(z);
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
                self.data.sigma.wise(&self.data.preactivation, &self.data.sigma, (struct {
                    pub fn func(z: dtype, sigma: dtype) dtype {
                        return sigma * options.activation.prime(z);
                    }
                }).func);
                loss = self.data.next_layer.layerBackprop(guess, correct);
            }
            // calculate weights gradient
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
                    return derivative * learning_rate;
                }
            }).func);
            self.data.weights.wise(&self.data.weight_gradients, &self.data.weights, (struct {
                pub fn func(weight: dtype, derivate: dtype) dtype {
                    return weight * -derivate;
                }
            }).func);
            if (comptime !is_final_layer) {
                self.data.next_layer.layerTrain(params);
            }
        }

        pub fn train(self: *@This(), input: anytype, correct: anytype, params: TrainingParams(dtype)) dtype {
            // const num_samples = input.shape[0];

            // const idx = params.random.uintLessThan(usize, num_samples);
            // var data_sample_tmp = input.mut(.{idx});
            // var data_sample = data_sample_tmp.reshape(.{ 1, data_sample_tmp.shape[0] });
            // var target_sample_tmp = correct.mut(.{idx});
            // var target_sample = target_sample_tmp.reshape(.{ 1, target_sample_tmp.shape[0] });
            // const loss = self.backprop(&data_sample, &target_sample);
            const loss = self.backprop(input, correct);
            self.layerTrain(params);
            return loss;
        }
    };
}
