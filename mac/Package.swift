// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Gall",
    platforms: [.macOS(.v13)],
    targets: [.executableTarget(name: "GallMenu", path: "Sources/Gall")]
)
