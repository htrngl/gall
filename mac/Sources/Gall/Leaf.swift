// the menu bar icon: a lobed oak leaf on an 18 pt grid, y down; the same
// outline is in mac/leaf.svg
import AppKit

let outline = """
    M 9 15.2 C 11 15.6 13 14 10.4 12.2 C 14.5 12.6 15.2 9.2 10.8 8.8 \
    C 15.4 8.4 15 4.6 10.6 5.4 C 13 3.4 11.4 1 9 1 \
    C 6.6 1 5 3.4 7.4 5.4 C 3 4.6 2.6 8.4 7.2 8.8 \
    C 2.8 9.2 3.5 12.6 7.6 12.2 C 5 14 7 15.6 9 15.2 Z
    """

// the M, C and Z of an svg path as an NSBezierPath
func path(_ d: String) -> NSBezierPath {
    let (b, t) = (NSBezierPath(), d.split(separator: " "))
    var i = 0
    func pt() -> NSPoint {
        i += 2
        return NSPoint(x: Double(t[i - 2])!, y: Double(t[i - 1])!)
    }
    while i < t.count {
        i += 1
        switch t[i - 1] {
        case "M": b.move(to: pt())
        case "C":
            let (c, e) = (pt(), pt())
            b.curve(to: pt(), controlPoint1: c, controlPoint2: e)
        default: b.close()
        }
    }
    return b
}

func leaf(_ alpha: Double) -> NSImage {
    let i = NSImage(size: NSSize(width: 18, height: 18), flipped: true) { _ in
        let g = NSGraphicsContext.current!.cgContext
        g.setAlpha(alpha) // dims the whole, with no darker overlaps
        g.beginTransparencyLayer(auxiliaryInfo: nil)
        NSColor.black.set()
        path(outline).fill()
        let rib = { (y0: Double, y1: Double, w: Double) in
            let b = NSBezierPath()
            b.move(to: NSPoint(x: 9, y: y0))
            b.line(to: NSPoint(x: 9, y: y1))
            (b.lineWidth, b.lineCapStyle) = (w, .round)
            b.stroke()
        }
        rib(15, 17.4, 1.2) // the stem
        NSGraphicsContext.current?.compositingOperation = .destinationOut
        rib(3.6, 14, 0.6) // the midrib, cut out
        g.endTransparencyLayer()
        return true
    }
    i.isTemplate = true
    return i
}
