// the menu bar app, or with "bridge" the telegram bridge alone
import Foundation

if CommandLine.arguments.dropFirst().first == "bridge" {
    MainActor.assumeIsolated {
        guard let b = Bridge() else { fatalError("no TELEGRAM_BOT_TOKEN") }
        b.start()
        bot = b
    }
    RunLoop.main.run()
} else {
    GallApp.main()
}
