// the menu bar: status, memory, the last chat, actions and settings
import AppKit
import ServiceManagement
import SwiftUI

func copy(_ s: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(s, forType: .string)
}

@main struct GallApp: App {
    @StateObject var m = Gall()

    var body: some Scene {
        MenuBarExtra {
            Menu(m: m)
        } label: {
            Image(nsImage: leaf(m.h?.loaded == true ? 1 : 0.35))
        }
        .menuBarExtraStyle(.menu)
    }
}

struct Menu: View {
    @ObservedObject var m: Gall

    // a setting in UserDefaults; a change restarts the server
    func pick<T>(_ k: String, _ v: T) -> Binding<T> {
        Binding(get: { m.d.object(forKey: k) as? T ?? v },
                set: { m.d.set($0, forKey: k); m.restart() })
    }

    var login: Binding<Bool> {
        Binding(get: { SMAppService.mainApp.status == .enabled }, set: {
            try? $0 ? SMAppService.mainApp.register()
                    : SMAppService.mainApp.unregister()
        })
    }

    var body: some View {
        Text(m.status)
        if let g = m.mem { Text(String(format: "Memory %.1f GB", g)) }
        if let l = m.last { Text(l) }
        Divider()
        Button("Copy Endpoint") { copy(endpoint + "/v1") }
        if m.net {
            Button("Copy Key") {
                copy((try? String(contentsOf: key))?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? "")
            }
        }
        Button("Unload Now") { m.park() }.disabled(m.h?.loaded != true)
        Button("Open Log") {
            let l = cache.appendingPathComponent("serve.log").path
            _ = try? Process.run(URL(fileURLWithPath: "/usr/bin/open"),
                                 arguments: ["-a", "Console", l])
        }
        SwiftUI.Menu("Settings") {
            Picker("Unload After", selection: pick("idle", 15)) {
                Text("5 min").tag(5)
                Text("15 min").tag(15)
                Text("60 min").tag(60)
                Text("Never").tag(0)
            }
            Picker("Context", selection: pick("ctx", 102400)) {
                Text("32k").tag(32768)
                Text("100k").tag(102400)
            }
            Toggle("Start at Login", isOn: login)
            Toggle("Allow Other Devices on the Network",
                   isOn: pick("net", false))
        }
        Divider()
        Button("Quit Gall") { m.quit() }
    }
}
