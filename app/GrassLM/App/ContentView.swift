import SwiftUI

/// Root view with sidebar chat history and main chat area.
struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var sidebarWidth: CGFloat = 240

    var body: some View {
        Group {
            if viewModel.isLoadingModel {
                loadingView
            } else if !viewModel.isModelLoaded {
                loadFailedView
            } else {
                mainLayout
            }
        }
        .ignoresSafeArea()
        .frame(minWidth: 600, minHeight: 400)
        .background(WindowConfigurator())
        .onAppear {
            viewModel.loadModel()
        }
    }

    // MARK: - Main Layout

    private var mainLayout: some View {
        HStack(spacing: 0) {
            SidebarView(viewModel: viewModel)
                .frame(width: sidebarWidth)
                .frame(maxHeight: .infinity)

            // Draggable divider
            Rectangle()
                .fill(Color.primary.opacity(0.08))
                .frame(width: 1)
                .contentShape(Rectangle().inset(by: -3))
                .onHover { hovering in
                    if hovering {
                        NSCursor.resizeLeftRight.push()
                    } else {
                        NSCursor.pop()
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 1, coordinateSpace: .global)
                        .onChanged { value in
                            sidebarWidth = max(160, min(360, value.location.x))
                        }
                )

            ChatView(viewModel: viewModel)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: - Loading State

    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
            Text("Loading GrassLM model...")
                .font(.headline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Error State

    private var loadFailedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.yellow)

            Text("Failed to Load Model")
                .font(.headline)

            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }

            Button("Retry") {
                viewModel.loadModel()
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Window Configuration

/// Configures the hosting NSWindow for a seamless, full-bleed title bar.
struct WindowConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let window = view.window {
                window.titlebarAppearsTransparent = true
                window.titleVisibility = .hidden
                window.styleMask.insert(.fullSizeContentView)
                window.isMovableByWindowBackground = true
                window.titlebarSeparatorStyle = .none
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
