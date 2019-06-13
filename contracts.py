from lazycontract import LazyContract, FloatProperty, \
IntegerProperty, StringProperty, ObjectProperty, ListProperty
from atom.contracts import BinaryProperty, RawContract, EmptyContract

INSTANCE_SEGMENTATION_ELEMENT_NAME = "instance-segmentation"

# Contracts for commands
class SegmentCommand:
    COMMAND_NAME = "segment"

    class Request(EmptyContract):
        SERIALIZE = False

    class Response(LazyContract):
        SERIALIZE = True

        rois = ListProperty(ListProperty(IntegerProperty()), required=True)
        scores = ListProperty(FloatProperty(), required=True)
        masks = ListProperty(BinaryProperty(), required=True)

class SetStreamCommand:
    COMMAND_NAME = "stream"

    # TODO: right now this is a binary string == "true" or "false",
    # we should change this and anyone who uses it to send a boolean property instead
    class Request(RawContract):
        SERIALIZE = False
        data = BinaryProperty(required=True)

    class Response(RawContract):
        SERIALIZE = False
        data = BinaryProperty(required=True)

class GetModeCommand:
    COMMAND_NAME = "get_mode"

    class Request(EmptyContract):
        SERIALIZE = False

    class Response(RawContract):
        SERIALIZE = False
        data = StringProperty(required=True)

class SetModeCommand:
    COMMAND_NAME = "set_mode"

    class Request(RawContract):
        SERIALIZE = False
        data = BinaryProperty(required=True)

    class Response(RawContract):
        SERIALIZE = False
        data = BinaryProperty(required=True)

# Contracts for publishing to streams
class ColorMaskStreamContract(LazyContract):
    STREAM_NAME = "color_mask"
    SERIALIZE = False

    data = BinaryProperty(required=True)
