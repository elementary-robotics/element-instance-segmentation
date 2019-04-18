## instance-segmentation

### Build Status

[![CircleCI](https://circleci.com/gh/elementary-robotics/element-instance-segmentation.svg?style=svg&circle-token=c758d0958636f52b65cce1196059c2995578fc31)](https://circleci.com/gh/elementary-robotics/element-instance-segmentation)

### Overview
instance-segmentation is an object agnostic foreground segmentation algorithm. It uses depth and grayscale data to determine if non-background objects are present in the image. If there are, then the algorithm will provide masks and bounding boxes for each object that it detects.
This element is based on [sd-maskrcnn](https://github.com/BerkeleyAutomation/sd-maskrcnn) visit their project page for more information on training or benchmarking a model.

### Building
Ensure that you have `git-lfs` installed, as the model weights are included in this repository.

Before running a build for the first time with `docker-compose`, initialize the submodules first.
```
git submodule update --init --recursive
```

### Commands
| Command  | Data              | Response |
| -------- | ----------------- | -------- |
| get_mode | None              | str      |
| set_mode | {"both", "depth"} | str      |
| segment  | None              | serialized dict of rois, scores, and TIF-encoded masks |
| stream   | {"true", "false"} | str      |

### Streams
To enable the streaming of `color_mask`, send `"true"` to the `stream` command.

| Stream                 | Format            |
| ---------------------- | ----------------- |
| color_mask             | TIF encoded image |



### docker-compose configuration
```
instance-segmentation:
  container_name: instance-segmentation
  build:
    context: .
    dockerfile: Dockerfile
  volumes:
    - type: volume
      source: shared
      target: /shared
      volume:
        nocopy: true
  depends_on:
    - "nucleus"
    - "realsense"
```
