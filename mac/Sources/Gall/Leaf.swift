// the menu bar icon: an oak leaf in outline, drawn upright on an 18 pt
// grid (y down) and turned 45 degrees; mac/leaf.svg draws the same
import AppKit

let outline = """
    M 9 1.2 C 11.3 1.335 12.185 2.685 10 3.9 C 13.9 4.06 14.105 5.66 \
    10.4 7.1 C 14.1 7.26 13.615 8.86 10.1 10.3 C 12.5 10.455 11.28 \
    12.005 9 13.4 C 6.72 12.005 5.5 10.455 7.9 10.3 C 4.385 8.86 3.9 \
    7.26 7.6 7.1 C 3.895 5.66 4.1 4.06 8 3.9 C 5.815 2.685 6.7 1.335 \
    9 1.2 Z
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
        case "L": b.line(to: pt())
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
        g.setAlpha(alpha) // dims the whole, with no darker crossings
        g.beginTransparencyLayer(auxiliaryInfo: nil)
        let t = NSAffineTransform()
        t.translateX(by: 8.2, yBy: 9.8) // centred once turned
        t.rotate(byDegrees: 45)
        t.translateX(by: -9, yBy: -9)
        t.concat()
        let b = path(outline + " M 9 16.6 L 9 4")
        (b.lineWidth, b.lineCapStyle, b.lineJoinStyle) = (0.85, .round, .round)
        NSColor.black.set()
        b.stroke()
        g.endTransparencyLayer()
        return true
    }
    i.isTemplate = true
    return i
}
