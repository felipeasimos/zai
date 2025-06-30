const std = @import("std");
const Tensor = @import("tensor").Tensor;
const contract = @import("contract");

pub fn TrainingParams(comptime dtype: type) type {
    return struct {
        learning_rate: dtype,
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
        next_layer_type: ?type,
    };
}

pub fn FullyConnectedLayer(comptime dtype: type, comptime options: FullyConnectedLayerOptions(dtype)) type {
    const weights_shape: [2]usize = .{ options.input_size, options.output_size };
    // const bias_shape = .{options.output_size};
    const input_shape = .{ options.batch_size, options.input_size };
    const output_shape: [2]usize = .{ options.batch_size, options.output_size };
    const is_final_layer = options.next_layer_type == null;

    const _final_output_shape = final_output_shape: {
        if (options.next_layer_type) |NextLayerType| {
            break :final_output_shape NextLayerType.final_output_shape;
        }
        break :final_output_shape output_shape;
    };

    const Data = data: {
        if (options.next_layer_type) |NextLayerType| {
            break :data struct {
                weights: Tensor(options.dtype, weights_shape),
                input: Tensor(options.dtype, input_shape),
                weight_gradients: Tensor(options.dtype, weights_shape),

                preactivation: Tensor(options.dtype, output_shape),
                sigma: Tensor(options.dtype, output_shape),

                next_layer: NextLayerType,
            };
        }
        break :data struct {
            weights: Tensor(options.dtype, weights_shape),
            input: Tensor(options.dtype, input_shape),
            weight_gradients: Tensor(options.dtype, weights_shape) = undefined,

            preactivation: Tensor(options.dtype, output_shape) = undefined,
            sigma: Tensor(options.dtype, output_shape) = undefined,

            guess: Tensor(options.dtype, output_shape) = undefined,
        };
    };

    return struct {
        data: Data,
        comptime final_output_shape: @TypeOf(_final_output_shape) = _final_output_shape,
        comptime final_output_type: type = Tensor(options.dtype, _final_output_shape),

        pub fn init() @This() {
            if (comptime @hasField(@This(), "next_layer")) {
                return .{
                    .weights = @FieldType(Data, "weights").random(),
                    .next_layer = @FieldType(Data, "next_layer").init(),
                };
            }
            return .{ .weights = @FieldType(@This(), "weights").random() };
        }

        // f(X * W + B)
        pub fn forward(self: *@This(), input: anytype, result: *@This().final_output_type) void {
            self.data.input.copy(input);
            var output = input.matmulNew(self.weights); // X * W
            self.data.preactivation.copy(output); // save Z
            output.apply(options.activation.f); // output = A = f(Z)

            if (comptime is_final_layer) {
                self.data.guess.copy(output);
                result.copy(output);
            } else {
                self.data.next_layer.forward(output, result);
            }
        }

        pub fn layerBackprop(self: *@This(), correct: anytype) void {
            // sigma[i] = dL/dZ[i+1] = (sigma[i+1] @ W[i+1]) * dA[i+1]/dZ[i+1]
            // if i == o: sigma = -(y-ŷ) * dA[i+1]/dZ[i+1]
            // dL/W[i] = A[i].T @ sigma[i]
            if (comptime is_final_layer) {
                // calculate sigma
                self.data.guess.wise(correct, &self.data.sigma, (struct {
                    pub fn func(a: dtype, b: dtype) dtype {
                        return a - b;
                    }
                }).func);
                self.data.sigma.wise(&self.data.preactivation, &self.data.sigma, (struct {
                    pub fn func(loss_gradient: dtype, z: dtype) dtype {
                        return loss_gradient * options.activation.prime(z);
                    }
                }));
            } else {
                self.data.next_layer.layerBackprop(self.data.input, correct);
                // calculate sigma
                self.data.next_layer.data.sigma.matmul(&self.data.next_layer.data.weights.transpose(.{}), &self.data.sigma);
                self.data.sigma.wise(&self.data.preactivation, &self.data.sigma, (struct {
                    pub fn func(z: dtype, sigma: dtype) dtype {
                        return sigma * options.activation.prime(z);
                    }
                }));
            }
            // calculate weights gradient
            self.data.input
                .transpose(.{})
                .matmul(&self.data.sigma, &self.data.weight_gradients);
        }

        // update bias: sum of current sigma
        // update weights: activations transposed and current sigma
        // sigma: activation gradient, next layer sigma and next layer weight transposed
        pub fn backprop(self: *@This(), input: anytype, correct: anytype) void {
            var result: @This().final_output_type = undefined;
            self.forward(input, &result);
            self.layerBackprop(correct);
        }

        pub fn layerTrain(self: *@This(), params: TrainingParams(dtype)) void {
            self.data.weight_gradients.wise(params.learning_rate, self.data.weight_gradients, (struct {
                pub fn func(derivative: dtype, learning_rate: dtype) dtype {
                    return derivative * learning_rate;
                }
            }).func);
            self.data.weights.wise(self.data.weight_gradients, self.data.weights, (struct {
                pub fn func(weight: dtype, derivate: dtype) dtype {
                    return weight * derivate;
                }
            }));
            if (!is_final_layer) {
                self.data.next_layer.layerTrain(params);
            }
        }

        pub fn train(self: *@This(), input: anytype, correct: anytype, params: TrainingParams(dtype)) void {
            self.backprop(input, correct);
            self.layerTrain(params);
        }
    };
}
