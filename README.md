### Overview
instance-segmentation is an object agnostic foreground segmentation algorithm. It uses depth and grayscale data to determine if non-background objects are present in the image. If there are, then the algorithm will provide masks and bounding boxes for each object that it detects.
This element is based on [sd-maskrcnn](https://github.com/BerkeleyAutomation/sd-maskrcnn) visit their project page for more information on training or benchmarking a model. 

### Commands
| Element Name            | Command Name           | Data              |
| ----------------------- | ---------------------- | ----------------- |
| instance-segmentation   | get_mode               | None              |
| instance-segmentation   | set_mode               | {"both", "depth"} |
| instance-segmentation   | segment                | None              |
