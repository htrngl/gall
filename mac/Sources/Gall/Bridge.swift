// the telegram bridge, in the app while its setting is on (alone: GallMenu
// bridge). a poller long-polls the bot and files each message from an
// allowed id in in/; a worker hands them, oldest first, to a coding agent;
// a sender drains out/ and keeps what telegram has not taken. so a quit or
// a dead network loses nothing. state is in ~/.cache/gall/bridge: offset
// (saved before an update is read: a message runs at most once), in/, out/
// and cont (the chat of the conversation); the log is ../bridge.log.
// ~/.config/gall/bridge (sh) sets TELEGRAM_BOT_TOKEN,
// TELEGRAM_ALLOWED_USERS (comma separated) and optionally BOT_DIR, TIMEOUT
// (seconds), GALL_URL, TELEGRAM_API and AGENT: a command for sh that reads
// the message on stdin and writes the answer on stdout. it runs in BOT_DIR,
// in its own process group, with GALL_URL, GALL_KEY, GALL_MODEL, CONTINUE
// (empty for a new conversation) and, for Resources/send, GALL_CHAT and
// GALL_OUT. the default, hax, also streams its tool calls to .jl for the
// draft. /new starts a conversation, /stop ends the answer that runs
import Foundation

struct No: Error { var wait: Double?, why: String } // no wait: for good

func log(_ s: String) {
    let p = cache.appendingPathComponent("bridge.log").path
    if !fm.fileExists(atPath: p) { fm.createFile(atPath: p, contents: nil) }
    guard let h = FileHandle(forWritingAtPath: p) else { return }
    h.seekToEndOfFile()
    h.write(Data("\(Date().formatted(.iso8601)) \(s)\n".utf8))
    try? h.close()
}

func text(_ u: URL) -> String {
    (try? String(contentsOf: u, encoding: .utf8)) ?? ""
}

func json(_ s: (any StringProtocol)?) -> [String: Any] {
    (try? JSONSerialization.jsonObject(with: Data((s.map { String($0) }
        ?? "").utf8))) as? [String: Any] ?? [:]
}

@MainActor final class Bridge {
    var c = ["TELEGRAM_API": "https://api.telegram.org", "TIMEOUT": "1800",
        "GALL_URL": endpoint + "/v1", "BOT_DIR": NSHomeDirectory() + "/Gall",
        "AGENT": "hax --json ${CONTINUE:+--continue} --provider=llamacpp "
            + "--model=$GALL_MODEL | tee .jl | jq -r 'select(.type == "
            + "\"result\") | .text // .error'"]
    let dir = cache.appendingPathComponent("bridge")
    let web = URLSession(configuration: .ephemeral)
    var jobs = [Task<Void, Never>](), long: Task<Any, Error>?
    var pid: pid_t = 0, why: String?, busy = false
    var work: URL { URL(fileURLWithPath: c["BOT_DIR"]!) }
    func at(_ s: String) -> URL { dir.appendingPathComponent(s) }

    init?() {
        // the file is sh: it is sourced, and sh prints what it set
        let (p, o) = (Process(), Pipe())
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", ". \"$0\"; for k in TELEGRAM_BOT_TOKEN "
            + "TELEGRAM_ALLOWED_USERS BOT_DIR TIMEOUT GALL_URL TELEGRAM_API "
            + "AGENT; do eval \"v=\\${$k-}\"; [ -z \"$v\" ] || "
            + "printf '%s=%s\\0' $k \"$v\"; done", conf.path]
        p.standardOutput = o
        try? p.run()
        let d = o.fileHandleForReading.readDataToEndOfFile()
        for l in String(decoding: d, as: UTF8.self).split(separator: "\0") {
            let p = l.split(separator: "=", maxSplits: 1,
                            omittingEmptySubsequences: false)
            c[String(p[0])] = String(p[1])
        }
        if c["TELEGRAM_BOT_TOKEN"] == nil { return nil }
    }

    func start() {
        for d in [at("in"), at("out"), work.appendingPathComponent("img")] {
            try? fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
        let (a, s) = (work.appendingPathComponent("AGENTS.md"),
                      "To send the user a file, run: send FILE [caption]\n")
        if !text(a).contains(s) {
            try? ((fm.fileExists(atPath: a.path) ? text(a) : "Telegram shows "
                + "your replies as plain text: never use markdown (no "
                + "asterisks, headings, backticks or tables).\n") + s)
                .write(to: a, atomically: true, encoding: .utf8)
        }
        jobs = [Task { await poll() }, Task { await run() },
                Task { await send() }]
    }

    // the files in in/ and out/ stay for the next start
    func stop() {
        jobs.forEach { $0.cancel() }
        long?.cancel()
        end("stopped")
    }

    // the long poll from before a sleep is dead
    func kick() { long?.cancel() }

    func end(_ s: String) {
        guard pid != 0 else { return }
        why = s
        kill(-pid, SIGTERM)
    }

    // a telegram method and its result; f is a file to upload, as field k
    @discardableResult
    func call(_ m: String, _ a: [String: Any] = [:], f: URL? = nil,
              k: String = "", t: Double = 30) async throws -> Any {
        var r = URLRequest(url: URL(string: c["TELEGRAM_API"]! + "/bot"
            + c["TELEGRAM_BOT_TOKEN"]! + "/" + m)!, timeoutInterval: t)
        r.httpMethod = "POST"
        if let f {
            let b = UUID().uuidString
            var d = Data()
            for (k, v) in a {
                d += Data(("--\(b)\r\ncontent-disposition: form-data; "
                    + "name=\"\(k)\"\r\n\r\n\(v)\r\n").utf8)
            }
            d += Data(("--\(b)\r\ncontent-disposition: form-data; name=\"\(k)"
                + "\"; filename=\"\(f.lastPathComponent)\"\r\n\r\n").utf8)
            d += try Data(contentsOf: f) + Data("\r\n--\(b)--\r\n".utf8)
            r.setValue("multipart/form-data; boundary=\(b)",
                       forHTTPHeaderField: "content-type")
            r.httpBody = d
        } else {
            r.setValue("application/json", forHTTPHeaderField: "content-type")
            r.httpBody = try JSONSerialization.data(withJSONObject: a)
        }
        let (d, _) = try await web.data(for: r)
        let j = json(String(decoding: d, as: UTF8.self))
        if j["ok"] as? Bool == true { return j["result"] ?? 0 }
        // telegram refuses a bad request for good; the rest is worth a wait
        let e = j["error_code"] as? Int ?? 500
        throw No(wait: (j["parameters"] as? [String: Any])?["retry_after"]
            as? Double ?? (e / 100 == 4 && e != 429 ? nil : 5),
            why: j["description"] as? String ?? "HTTP \(e)")
    }

    // a file in a queue; the oldest by its time of writing goes first
    func put(_ q: String, _ s: String) {
        try? s.write(to: at("\(q)/\(UUID().uuidString).text"),
                     atomically: true, encoding: .utf8)
    }

    func queue(_ q: String) -> URL? {
        let k = URLResourceKey.contentModificationDateKey
        let t = { (u: URL) in
            (try? u.resourceValues(forKeys: [k]))?.contentModificationDate
                ?? .distantFuture
        }
        return ((try? fm.contentsOfDirectory(at: at(q),
            includingPropertiesForKeys: [k])) ?? [])
            .filter { ["text", "file"].contains($0.pathExtension) }
            .min { t($0) < t($1) }
    }

    // telegram takes 4096 UTF-16 units a message
    func say(_ chat: String, _ s: String) {
        let s = Array(s.isEmpty ? "(no output)" : s)
        for i in stride(from: 0, to: s.count, by: 3500) {
            put("out", chat + "\n" + String(s[i..<min(i + 3500, s.count)]))
        }
    }

    func poll() async {
        _ = try? await call("setMyCommands", ["commands": [
            ["command": "new", "description": "start a new conversation"],
            ["command": "stop", "description": "stop the answer that runs"]]])
        let ok = Set((c["TELEGRAM_ALLOWED_USERS"] ?? "").split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) })
        while !Task.isCancelled {
            let o = Int(text(at("offset"))) ?? 0
            long = Task {
                try await call("getUpdates", ["timeout": 60, "offset": o],
                               t: 70)
            }
            guard let us = try? await long!.value as? [[String: Any]] else {
                await sleep(2)
                continue
            }
            for u in us {
                try? "\((u["update_id"] as? Int ?? o) + 1)".write(
                    to: at("offset"), atomically: true, encoding: .utf8)
                guard let m = u["message"] as? [String: Any],
                      let ch = (m["chat"] as? [String: Any])?["id"],
                      let f = (m["from"] as? [String: Any])?["id"]
                else { continue }
                guard ok.contains("\(f)") else {
                    log("ignored a message from \(f)")
                    continue
                }
                await take(m, "\(ch)")
            }
        }
    }

    // a photo lands in img/ and the model looks at it with its read tool
    func take(_ m: [String: Any], _ chat: String) async {
        var t = m["text"] as? String ?? m["caption"] as? String ?? ""
        if t == "/new" {
            try? fm.removeItem(at: at("cont"))
            return say(chat, "new conversation")
        }
        if t == "/stop" {
            return pid == 0 ? say(chat, "nothing runs") : end("stopped")
        }
        if let p = (m["photo"] as? [[String: Any]])?.last?["file_id"] {
            guard let r = try? await call("getFile", ["file_id": p]),
                  let p = (r as? [String: Any])?["file_path"] as? String,
                  let u = URL(string: c["TELEGRAM_API"]! + "/file/bot"
                      + c["TELEGRAM_BOT_TOKEN"]! + "/" + p),
                  let (d, _) = try? await web.data(from: u)
            else { return say(chat, "could not fetch the image") }
            let f = work.appendingPathComponent("img/" + u.lastPathComponent)
            try? d.write(to: f)
            t += "\n(the user sent an image: \(f.path). read it to see it.)"
        } else if m["text"] == nil {
            return say(chat, "only text and photos can be read")
        }
        put("in", chat + "\n" + t)
    }

    func run() async {
        while !Task.isCancelled {
            guard let f = queue("in") else {
                await sleep(0.5)
                continue
            }
            let p = text(f).split(separator: "\n", maxSplits: 1,
                                  omittingEmptySubsequences: false)
            try? fm.removeItem(at: f)
            guard p.count == 2 else { continue }
            busy = true
            await answer(String(p[0]), String(p[1]))
            busy = false
        }
    }

    func up() async -> Bool {
        let u = URL(string: c["GALL_URL"]!.replacingOccurrences(
            of: "/v1", with: "/health"))!
        return (try? await web.data(for: URLRequest(url: u,
            timeoutInterval: 5))) != nil
    }

    func answer(_ chat: String, _ t: String) async {
        log("\(chat): \(t)")
        // the server restarts when a setting changes: wait for it
        var i = 0
        while i < 60, !(await up()) {
            i += 1
            await sleep(2)
        }
        var e = ProcessInfo.processInfo.environment
        let k = text(key).trimmingCharacters(in: .whitespacesAndNewlines)
        let x = UserDefaults.standard.persistentDomain(
            forName: "io.github.ltrgoddard.gall")?["ctx"] ?? 102400
        e.merge(["GALL_URL": c["GALL_URL"]!, "GALL_MODEL": "inkling-2-mini",
            "GALL_KEY": k.isEmpty ? "none" : k, "GALL_CHAT": chat,
            "GALL_OUT": at("out").path,
            "CONTINUE": text(at("cont")) == chat ? "1" : "",
            "PATH": (Bundle.main.resourcePath ?? ".") + ":" + (e["PATH"] ?? "")
                + ":/opt/homebrew/bin:/usr/local/bin",
            // hax as a llama.cpp client of gall, compacting before the
            // app's window fills
            "HAX_LLAMACPP_BASE_URL": c["GALL_URL"]!,
            "HAX_LLAMACPP_API_KEY": k.isEmpty ? "none" : k,
            "HAX_CONTEXT_LIMIT": "\(x)", "HAX_MAX_TURNS": "0",
            "HAX_IMAGE_INPUT": "on"]) { $1 }
        try? fm.removeItem(at: work.appendingPathComponent(".jl"))
        try? t.write(to: at("msg"), atomically: true, encoding: .utf8)
        why = nil
        pid = spawn(c["AGENT"]!, e, work.path,
                    [at("msg").path, at("ans").path, at("err").path])
        // while the agent runs, a draft (gone with the answer) shows the
        // tool it ran last; a draft is typed out again when it changes, so
        // send a new one, or each 20 s
        let w = Task {
            var (n, y) = (0, "-")
            while true {
                await sleep(4)
                n += 4
                guard !Task.isCancelled else { break }
                let x = tool()
                if x != y || n % 20 == 0 {
                    y = x
                    _ = try? await call("sendMessageDraft", ["chat_id": chat,
                        "draft_id": 1, "text": x.isEmpty ? "working" : x],
                        t: 5)
                }
                if n >= Int(c["TIMEOUT"]!) ?? 1800 { end("timed out") }
            }
        }
        if pid == 0 { why = "could not start the agent in \(work.path)" }
        let s = pid == 0 ? 0 : await wait(pid)
        w.cancel()
        pid = 0
        try? chat.write(to: at("cont"), atomically: true, encoding: .utf8)
        say(chat, why ?? (s == 0 ? text(at("ans")) : text(at("ans"))
            + text(at("err")).split(separator: "\n")
            .filter { !$0.hasPrefix("resume with") }.suffix(4)
            .joined(separator: "\n"))
            .trimmingCharacters(in: .whitespacesAndNewlines))
    }

    // "$ git status …", "read README.md": a command's first line, without
    // a leading cd, or a file's name, cut at a word near 40 characters
    func tool() -> String {
        let j = json(text(work.appendingPathComponent(".jl"))
            .split(separator: "\n")
            .last { $0.contains("\"kind\":\"tool_call\"") })
        guard let t = j["tool_name"] as? String else { return "" }
        let a = json(j["arguments"] as? String)
        var s = a["command"] as? String ?? (a["path"] as? String)
            .map { ($0 as NSString).lastPathComponent } ?? ""
        let re = { (p: String, w: String) in
            s = s.replacingOccurrences(of: p, with: w,
                                       options: .regularExpression)
        }
        s = String(s.split(separator: "\n").first ?? "")
        re("^ *cd [^ ]+ *(&&|;) *", "")
        re("\\s+", " ")
        if s.count > 40 {
            s = String(s.prefix(40))
            re(" [^ ]*$", " …")
        }
        return s.isEmpty ? t : t == "bash" ? "$ " + s : t + " " + s
    }

    // an entry of out/ is a chat id and a text, or (.file, from
    // Resources/send) a chat id, a path and a caption
    func send() async {
        var back = 1.0
        while !Task.isCancelled {
            guard let f = queue("out") else {
                await sleep(0.5)
                continue
            }
            let file = f.pathExtension == "file"
            let p = text(f).split(separator: "\n", maxSplits: file ? 2 : 1,
                omittingEmptySubsequences: false).map { String($0) }
            do {
                if file, p.count == 3 {
                    let u = URL(fileURLWithPath: p[1])
                    let n = (try? u.resourceValues(forKeys: [.fileSizeKey]))?
                        .fileSize ?? 0
                    let k = ["png", "jpg", "jpeg", "webp"].contains(
                        u.pathExtension.lowercased()) && n < 10_000_000
                        ? "photo" : "document"
                    try await call("send" + k.capitalized, ["chat_id": p[0],
                        "caption": p[2]], f: u, k: k, t: 300)
                } else if !file, p.count == 2 {
                    try await call("sendMessage", ["chat_id": p[0],
                                                   "text": p[1]])
                }
                back = 1
            } catch let e as No where e.wait == nil {
                log("dropped \(f.lastPathComponent): \(e.why)")
                if file { say(p[0], "could not send \(p[1]): \(e.why)") }
            } catch {
                await sleep((error as? No)?.wait ?? back)
                back = min(back * 2, 60)
                continue
            }
            try? fm.removeItem(at: f)
        }
    }
}

// sh -c cmd in its own process group, so a stop reaches all of it, with
// stdin, stdout and stderr on the files io and no other descriptor
func spawn(_ cmd: String, _ env: [String: String], _ cwd: String,
           _ io: [String]) -> pid_t {
    var (fa, at): (posix_spawn_file_actions_t?, posix_spawnattr_t?)
    posix_spawn_file_actions_init(&fa)
    posix_spawnattr_init(&at)
    posix_spawn_file_actions_addchdir_np(&fa, cwd)
    for (i, p) in io.enumerated() {
        posix_spawn_file_actions_addopen(&fa, Int32(i), p, i == 0 ? O_RDONLY
            : O_WRONLY | O_CREAT | O_TRUNC, 0o600)
    }
    posix_spawnattr_setflags(&at, Int16(POSIX_SPAWN_SETPGROUP
        | POSIX_SPAWN_CLOEXEC_DEFAULT))
    let dup = { (s: String) -> UnsafeMutablePointer<CChar>? in strdup(s) }
    let a = ["/bin/sh", "-c", cmd].map(dup) + [nil]
    let e = env.map { dup($0 + "=" + $1) } + [nil]
    var pid: pid_t = 0
    posix_spawn(&pid, "/bin/sh", &fa, &at, a, e)
    (a + e).forEach { free($0) }
    posix_spawn_file_actions_destroy(&fa)
    posix_spawnattr_destroy(&at)
    return pid
}

func wait(_ p: pid_t) async -> Int32 {
    await withCheckedContinuation { k in
        DispatchQueue.global().async {
            var s: Int32 = 0
            waitpid(p, &s, 0)
            k.resume(returning: s)
        }
    }
}
