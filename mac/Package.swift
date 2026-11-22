// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Gall",
    platforms: [.macOS("15.0")],
    targets: [.executableTarget(name: "GallMenu", path: "Sources/Gall")]
)
