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
            Image(nsImage: leaf(m.h.map { $0.loaded || $0.busy == true }
                                == true))
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
        if let g = m.mem {
            Text(g < 1 ? String(format: "Memory %.0f MB", g * 1000)
                       : String(format: "Memory %.1f GB", g))
        }
        if let l = m.last { Text(l) }
        Divider()
        Button("Copy endpoint") { copy(endpoint + "/v1") }
        if m.net {
            Button("Copy key") {
                copy((try? String(contentsOf: key, encoding: .utf8))?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? "")
            }
        }
        Button("Unload now") { m.park() }.disabled(m.h?.loaded != true)
        Button("Open log") {
            let l = cache.appendingPathComponent("serve.log").path
            _ = try? Process.run(URL(fileURLWithPath: "/usr/bin/open"),
                                 arguments: ["-a", "Console", l])
        }
        SwiftUI.Menu("Settings") {
            // seconds idle; instant is 1 s after the answer, 0 is never
            Picker("Unload after", selection: pick("unload", 1)) {
                Text("Instant").tag(1)
                Text("1 min").tag(60)
                Text("5 min").tag(300)
                Text("15 min").tag(900)
                Text("60 min").tag(3600)
                Text("Never").tag(0)
            }
            Picker("Context", selection: pick("ctx", 102400)) {
                Text("32k").tag(32768)
                Text("100k").tag(102400)
            }
            Toggle("Start at login", isOn: login)
            Toggle("Allow other devices on the network",
                   isOn: pick("net", false))
            Toggle("Telegram bridge", isOn: $m.tg)
            Toggle("Wake every 5 minutes", isOn: $m.wake).disabled(!m.tg)
        }
        Divider()
        Button("Quit Gall") { m.quit() }
    }
}
