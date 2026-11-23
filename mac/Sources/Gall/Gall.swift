// the state of the server and the commands that change it: polls /health,
// reads the footprint of the server pid, runs the bundled gall binary
import AppKit
import IOKit.pwr_mgt

let fm = FileManager.default
let cache = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/gall")
let key = cache.appendingPathComponent("key")
let endpoint = "http://127.0.0.1:9940"
let conf = fm.homeDirectoryForCurrentUser
    .appendingPathComponent(".config/gall/bridge")

// the telegram bridge (Resources/bridge) while its setting is on; its
// process group goes with it: hax and curl
var bot: Process?
func unbot() {
    if let p = bot, p.isRunning { kill(-p.processIdentifier, SIGTERM) }
    bot = nil
    lid(false)
}

// no sleep while the bridge runs: an idle-sleep assertion, which the
// system drops if we die, and for a closed lid the root domain's
// kPMSetClamshellSleepState (12), which needs no root. the kernel keeps
// the bit after we exit, so every start of the app clears it (bridge()
// calls unbot), and the poll sets it again if powerd clears it
let pm: io_connect_t = {
    var c: io_connect_t = 0
    IOServiceOpen(IOServiceGetMatchingService(kIOMainPortDefault,
        IOServiceMatching("IOPMrootDomain")), mach_task_self_, 0, &c)
    return c
}()
var nap: IOPMAssertionID = 0
func lid(_ on: Bool) {
    var v: UInt64 = on ? 1 : 0
    if on, nap == 0 {
        IOPMAssertionCreateWithName(
            kIOPMAssertPreventUserIdleSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            "Gall Telegram bridge" as CFString, &nap)
    } else if !on, nap != 0 { IOPMAssertionRelease(nap); nap = 0 }
    IOConnectCallScalarMethod(pm, 12, &v, 1, nil, nil)
}

struct Health: Decodable, Equatable {
    var loaded: Bool
    var busy: Bool?
    var last: Last?
    struct Last: Decodable, Equatable {
        var cacheN, promptN: Int?
        var predictedPerSecond: Double?
    }
}

// Contents/MacOS/gall, else a development build or gall on PATH; never the
// app itself (Gall and gall are one file on a case-insensitive volume)
let bin: URL? = {
    let me = Bundle.main.executableURL!.resolvingSymlinksInPath()
    let ino = { (u: URL) in
        (try? fm.attributesOfItem(atPath: u.path))?[.systemFileNumber] as? Int
    }
    let up = { (u: URL, n: Int) in
        (0..<n).reduce(u) { u, _ in u.deletingLastPathComponent() }
    }
    let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
    let c = [up(me, 1), up(me, 5).appendingPathComponent("target/release"),
             URL(fileURLWithPath: fm.currentDirectoryPath + "/target/release")]
        + path.split(separator: ":").map { URL(fileURLWithPath: String($0)) }
    return c.map { $0.appendingPathComponent("gall") }.first {
        fm.isExecutableFile(atPath: $0.path) && ino($0) != ino(me)
    }
}()

// exit status and output of gall with these arguments
@discardableResult
func gall(_ a: [String]) async -> (Int32, String) {
    guard let bin else { return (-1, "The gall binary is missing.") }
    return await withCheckedContinuation { k in
        // read as it runs: curl fills the pipe during a download
        DispatchQueue.global().async {
            let (p, o) = (Process(), Pipe())
            (p.executableURL, p.arguments) = (bin, a)
            (p.standardOutput, p.standardError) = (o, o)
            do { try p.run() } catch {
                return k.resume(returning: (-1, "\(error)"))
            }
            try? o.fileHandleForWriting.close()
            let d = o.fileHandleForReading.readDataToEndOfFile().suffix(65536)
            p.waitUntilExit()
            k.resume(returning: (p.terminationStatus,
                                 String(decoding: d, as: UTF8.self)))
        }
    }
}

func sleep(_ s: Double) async {
    try? await Task.sleep(nanoseconds: UInt64(s * 1e9))
}

// bytes in ~/.cache/gall of the model files, finished or partial
func got() -> Double {
    let f = (try? fm.contentsOfDirectory(at: cache,
        includingPropertiesForKeys: [.fileSizeKey])) ?? []
    return f.filter { ["gguf", "part"].contains($0.pathExtension) }
        .compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }
        .reduce(0) { $0 + Double($1) }
}

// physical footprint of the server in GB (1e9 B), as activity monitor has it
func footprint() -> Double? {
    let s = try? String(contentsOf: cache.appendingPathComponent("serve.pid"),
                        encoding: .utf8)
    guard let p = Int32(s?.trimmingCharacters(in: .whitespacesAndNewlines)
                        ?? "") else { return nil }
    var ri = rusage_info_v4()
    let r = withUnsafeMutablePointer(to: &ri) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(p, RUSAGE_INFO_V4, $0)
        }
    }
    return r == 0 ? Double(ri.ri_phys_footprint) / 1e9 : nil
}

func alert(_ s: String) {
    NSApp.activate(ignoringOtherApps: true)
    let a = NSAlert()
    a.messageText = s
    a.runModal()
}

@MainActor final class Gall: ObservableObject {
    @Published var h: Health?
    @Published var phase = "" // overrides the status while not empty
    @Published var mem: Double?
    var fetching = false
    let d = UserDefaults.standard

    var idle: Int { d.object(forKey: "unload") as? Int ?? 1 } // seconds
    var ctx: Int { d.object(forKey: "ctx") as? Int ?? 102400 }
    var net: Bool { d.bool(forKey: "net") }
    // published, so the menu's tick follows a click
    @Published var tg = UserDefaults.standard.bool(forKey: "bridge") {
        didSet { d.set(tg, forKey: "bridge"); bridge() }
    }

    var status: String {
        guard phase.isEmpty else { return phase }
        guard let h else { return "Stopped" }
        return h.busy == true ? "Answering" : h.loaded ? "Ready" : "Unloaded"
    }

    var last: String? {
        guard let t = h?.last, let c = t.cacheN, let p = t.promptN
        else { return nil }
        let r = String(format: "%.1f", t.predictedPerSecond ?? 0)
        return "Last: \(c.formatted()) of \((c + p).formatted()) tokens "
            + "cached, \(r) t/s"
    }

    // a poll a second: the icon follows a load or an unload at once
    init() {
        // every quit stops the server: Quit Gall, a logout, or a quit from
        // outside the menu (which an update of the app needs)
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil,
            queue: nil) { _ in
            unbot()
            try? bin.map { try Process.run($0, arguments: ["down"]) }?
                .waitUntilExit()
        }
        Task {
            NSApp.setActivationPolicy(.accessory)
            await start()
            bridge()
            while true {
                if fetching {
                    let p = "Downloading \(min(100, Int(got() / 12.7e7)))%"
                    if p != phase { phase = p }
                }
                if let b = bot { b.isRunning ? lid(true) : unbot() }
                await poll()
                await sleep(1)
            }
        }
    }

    let session = URLSession(configuration: .ephemeral)

    func health() async -> Health? {
        let u = URL(string: endpoint + "/health")!
        let r = URLRequest(url: u, timeoutInterval: 2)
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        guard let (b, _) = try? await session.data(for: r)
        else { return nil }
        return try? dec.decode(Health.self, from: b)
    }

    // only a change is published: each one rebuilds the open menu, which
    // closes a submenu under the pointer
    func poll() async {
        let n = await health()
        let g = n == nil ? nil : footprint().map {
            let k = $0 < 1 ? 1000.0 : 10.0 // MB below 1 GB
            return ($0 * k).rounded() / k
        }
        if n != h { h = n }
        if g != mem { mem = g }
    }

    // the first run checks the mac and fetches the model; up is a no-op
    // when a server answers already
    func start() async {
        if await health() != nil { return await poll() }
        try? fm.createDirectory(at: cache, withIntermediateDirectories: true)
        if !d.bool(forKey: "installed") {
            let ram = ProcessInfo.processInfo.physicalMemory >> 30
            let free = (try? cache.resourceValues(forKeys:
                [.volumeAvailableCapacityForImportantUsageKey]))?
                .volumeAvailableCapacityForImportantUsage ?? 0
            let need = (14e9 - got()) / 1e9
            let why = ram < 24 ? "Gall needs 24 GB of memory. This Mac has "
                + "\(ram) GB." : Double(free) / 1e9 < need ? "Gall needs "
                + String(format: "%.1f", need) + " GB free on the home "
                + "volume for the model." : nil
            if let why {
                phase = "Not installed"
                return alert(why)
            }
        }
        fetching = true
        let (s, o) = await gall(["model"])
        fetching = false
        guard s == 0 else {
            phase = "Download failed"
            return alert(o)
        }
        d.set(true, forKey: "installed")
        await up()
    }

    func up() async {
        phase = "Starting"
        var a = ["up", String(ctx), "--idle-unload", String(idle)]
        if net {
            if !fm.fileExists(atPath: key.path) {
                let k = (0..<32).map { _ in
                    String(format: "%02x", UInt8.random(in: 0...255))
                }.joined()
                fm.createFile(atPath: key.path, contents: Data(k.utf8),
                              attributes: [.posixPermissions: 0o600])
            }
            a += ["--host", "0.0.0.0", "--api-key-file", key.path]
        }
        let (s, o) = await gall(a)
        phase = s == 0 ? "" : "Failed to start"
        if s != 0 { alert(o) }
        await poll()
    }

    // down parks the conversation; wait for the port to close, then up
    func restart() {
        Task {
            phase = "Restarting"
            await gall(["down"])
            for _ in 0..<600 {
                guard await health() != nil else { break }
                await sleep(0.2)
            }
            await up()
        }
    }

    func park() {
        Task {
            var r = URLRequest(url: URL(string: endpoint + "/park")!)
            r.httpMethod = "POST"
            if net, let k = try? String(contentsOf: key, encoding: .utf8) {
                let k = k.trimmingCharacters(in: .whitespacesAndNewlines)
                r.setValue("Bearer " + k, forHTTPHeaderField: "Authorization")
            }
            _ = try? await session.data(for: r)
            await poll()
        }
    }

    // log in ~/.cache/gall/bridge.log
    func bridge() {
        unbot()
        guard tg, let s = Bundle.main.url(forResource: "bridge",
                                          withExtension: nil) else { return }
        guard fm.fileExists(atPath: conf.path) else {
            return alert("Put TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USERS "
                + "(your numeric ID) in ~/.config/gall/bridge.")
        }
        let l = cache.appendingPathComponent("bridge.log").path
        if !fm.fileExists(atPath: l) { fm.createFile(atPath: l, contents: nil) }
        let (p, o) = (Process(), FileHandle(forWritingAtPath: l))
        o?.seekToEndOfFile()
        (p.executableURL, p.standardOutput, p.standardError) = (s, o, o)
        try? p.run()
        bot = p
    }

    func quit() { NSApp.terminate(nil) }
}
