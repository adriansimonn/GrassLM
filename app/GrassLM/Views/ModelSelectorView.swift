import SwiftUI

/// Available model definitions that can be selected.
struct ModelInfo: Identifiable, Hashable {
    let id: String
    let displayName: String
    let parameterCount: String
    let resourceName: String   // Bundle resource name (without extension)
    let fileExtension: String  // Bundle resource extension

    static let available: [ModelInfo] = [
        ModelInfo(
            id: "grasslm-10m",
            displayName: "GrassLM 10M",
            parameterCount: "12.6M params",
            resourceName: "grasslm-6L",
            fileExtension: "grasslm"
        )
        // Additional models can be added here as they become available
    ]
}

/// Dropdown model selector shown in the chat header area.
struct ModelSelectorView: View {
    @Binding var selectedModelID: String
    var isModelLoaded: Bool
    var isLoadingModel: Bool

    var body: some View {
        Menu {
            ForEach(ModelInfo.available) { model in
                Button {
                    selectedModelID = model.id
                } label: {
                    HStack {
                        VStack(alignment: .leading) {
                            Text(model.displayName)
                            Text(model.parameterCount)
                                .font(.caption2)
                        }
                        if selectedModelID == model.id {
                            Spacer()
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                if isLoadingModel {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Circle()
                        .fill(isModelLoaded ? Color.green : Color.orange)
                        .frame(width: 7, height: 7)
                }

                Text(selectedModelDisplayName)
                    .font(.subheadline)
                    .fontWeight(.medium)

            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.primary.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    private var selectedModelDisplayName: String {
        ModelInfo.available.first(where: { $0.id == selectedModelID })?.displayName ?? "Select Model"
    }
}
