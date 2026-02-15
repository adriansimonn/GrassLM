import SwiftUI

/// Main chat interface with centered content layout, model selector, and input bar.
struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: 0) {
            // Messages area or empty state
            if viewModel.messages.isEmpty {
                emptyState
            } else {
                messageList
            }

            // Input bar at bottom
            inputBar
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()

            Image(systemName: "leaf.fill")
                .font(.system(size: 36))
                .foregroundStyle(.green.opacity(0.6))

            Text("GrassLM")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundStyle(.primary)

            Text("Start a conversation")
                .font(.subheadline)
                .foregroundStyle(.tertiary)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Message List

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(viewModel.messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }

                    if viewModel.isGenerating {
                        HStack {
                            TypingIndicator()
                                .padding(.leading, 20)
                            Spacer()
                        }
                        .id("typing-indicator")
                    }
                }
                .padding(.vertical, 16)
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: viewModel.messages.count) {
                scrollToBottom(proxy: proxy)
            }
            .onChange(of: viewModel.messages.last?.content) {
                scrollToBottom(proxy: proxy)
            }
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 0) {
            // Error banner
            if let error = viewModel.errorMessage {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                        .font(.caption)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Dismiss") {
                        viewModel.errorMessage = nil
                    }
                    .buttonStyle(.plain)
                    .font(.caption)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }

            // Prompt box
            VStack(spacing: 0) {
                // Text field
                TextField("Message GrassLM...", text: $viewModel.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...8)
                    .onSubmit {
                        if NSApp.currentEvent?.modifierFlags.contains(.shift) == false {
                            viewModel.send()
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.top, 14)
                    .padding(.bottom, 14)

                Divider()
                    .opacity(0.3)
                    .padding(.horizontal, 10)

                // Bottom controls row: model selector + settings on left, send on right
                HStack(spacing: 8) {
                    ModelSelectorView(
                        selectedModelID: $viewModel.selectedModelID,
                        isModelLoaded: viewModel.isModelLoaded,
                        isLoadingModel: viewModel.isLoadingModel
                    )

                    Button {
                        showSettings.toggle()
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(.primary.opacity(0.7))
                            .frame(width: 28, height: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Generation Settings")
                    .popover(isPresented: $showSettings) {
                        SettingsView(viewModel: viewModel)
                    }

                    Spacer()

                    if viewModel.isGenerating {
                        Button {
                            viewModel.cancelGeneration()
                        } label: {
                            Image(systemName: "stop.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.red)
                        }
                        .buttonStyle(.plain)
                        .help("Stop generating")
                    } else {
                        Button {
                            viewModel.send()
                        } label: {
                            let canSend = !viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && viewModel.isModelLoaded
                            Image(systemName: "arrow.up")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundStyle(.white)
                                .frame(width: 26, height: 26)
                                .background(
                                    RoundedRectangle(cornerRadius: 6)
                                        .fill(canSend ? Color.green : Color.gray)
                                )
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !viewModel.isModelLoaded)
                        .help("Send (Enter)")
                    }
                }
                .padding(.horizontal, 10)
                .padding(.top, 8)
                .padding(.bottom, 10)
            }
            .background(Color(.controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(Color.primary.opacity(0.1), lineWidth: 1)
            )
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
            .padding(.top, 8)
        }
        .frame(maxWidth: 720)
        .frame(maxWidth: .infinity)
    }

    // MARK: - Helpers

    private func scrollToBottom(proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.2)) {
            if viewModel.isGenerating {
                proxy.scrollTo("typing-indicator", anchor: .bottom)
            } else if let lastID = viewModel.messages.last?.id {
                proxy.scrollTo(lastID, anchor: .bottom)
            }
        }
    }
}

// MARK: - Typing Indicator

/// Animated dots shown while the model is generating.
struct TypingIndicator: View {
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { index in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 6, height: 6)
                    .opacity(phase == index ? 1.0 : 0.3)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { _ in
                withAnimation(.easeInOut(duration: 0.3)) {
                    phase = (phase + 1) % 3
                }
            }
        }
    }
}
