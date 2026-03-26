// Place in scala2d directory

package scalation.scala2d

import java.awt.Component
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/**
 * Saves a graphical component (such as a plot) to a PNG file.
 *
 * This method renders the given AWT `Component` into a `BufferedImage` and writes it
 * to the specified file path. It includes a scaling factor to increase the resolution 
 * of the output image, which is especially useful for rendering crisp text and graphics 
 * on high-DPI displays.
 *
 * @param filename  The path and name of the file where the image will be saved (e.g., "output/plot.png").
 * @param component The AWT Component (e.g., a panel or frame containing the plot) to be saved.
 * @param scale     The scaling multiplier to apply for higher resolution rendering (default is 2.0).
 */
def savePlot(filename: String, component: Component, scale: Double = 2.0): Unit =
  
  // Pause execution briefly to ensure the component has finished its layout and rendering lifecycle.
  // Note: While Thread.sleep is a blunt tool, it's often necessary for AWT/Swing components 
  // to fully paint before attempting to capture them.
  Thread.sleep(1000) 

  // Calculate the scaled dimensions for the output image based on the component's actual size
  val width = (component.getWidth * scale).toInt
  val height = (component.getHeight * scale).toInt

  // Create a high-resolution image buffer with standard RGB color space
  val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
  val g2d = image.createGraphics()

  // Apply the scale transformation so the component paints at the higher resolution
  g2d.scale(scale, scale)

  // Instruct the component to paint itself onto our scaled Graphics2D context
  component.paint(g2d)
  
  // Dispose of the graphics context to free up system resources
  g2d.dispose()
  
  // Write the buffered image to disk as a PNG file
  ImageIO.write(image, "png", new File(filename))
  println(s"Plot saved to: $filename")
end savePlot