import SwiftUI

/// Settings panel for controlling generation parameters.
struct SettingsView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Generation Settings")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding()

            Divider()

            Form {
                // Temperature
                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Temperature")
                            Spacer()
                            Text(String(format: "%.2f", viewModel.temperature))
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Slider(value: $viewModel.temperature, in: 0...2, step: 0.05)
                        Text("Lower values produce more focused output. Higher values increase randomness.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Top-P (Nucleus Sampling)
                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Top-P")
                            Spacer()
                            Text(String(format: "%.2f", viewModel.topP))
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Slider(value: $viewModel.topP, in: 0.1...1.0, step: 0.05)
                        Text("Nucleus sampling threshold. Lower values restrict to higher-probability tokens.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Max Tokens
                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Max Tokens")
                            Spacer()
                            Text("\(viewModel.maxTokens)")
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Slider(
                            value: Binding(
                                get: { Double(viewModel.maxTokens) },
                                set: { viewModel.maxTokens = Int($0) }
                            ),
                            in: 16...512,
                            step: 16
                        )
                        Text("Maximum number of tokens to generate per response.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Reset
                Section {
                    Button("Reset to Defaults") {
                        viewModel.temperature = 0.8
                        viewModel.topP = 0.9
                        viewModel.maxTokens = 128
                    }
                }
            }
            .formStyle(.grouped)
        }
        .frame(width: 400, height: 420)
    }
}
