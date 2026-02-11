import SwiftUI

/// Root view that manages model loading and presents the chat interface.
struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()

    var body: some View {
        Group {
            if viewModel.isLoadingModel {
                loadingView
            } else if !viewModel.isModelLoaded {
                loadFailedView
            } else {
                ChatView(viewModel: viewModel)
            }
        }
        .frame(minWidth: 400, minHeight: 300)
        .navigationTitle("GrassLM")
        .onAppear {
            viewModel.loadModel()
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
