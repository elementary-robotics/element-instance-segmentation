from lazycontract import LazyContract, LazyProperty, FloatProperty, \
IntegerProperty, StringProperty, ObjectProperty, ListProperty

class BinaryProperty(LazyProperty):
    _type = bytes
    def deserialize(self, obj):
        return obj if isinstance(obj, self._type) else LazyContractDeserializationError("Must provide bytes object")

class RawContract(LazyProperty):
    def __init__(self,  *args, **kwargs):
        super(RawContract, self).__init__(*args, **kwargs)
        if 'data' not in self._properties:
            raise LazyContractValidationError("Raw contracts must specify a data field name and type")

    def to_dict(self):
        raise TypeError("Cannot convert raw contract to dict, use the to_data() function instead")

    def to_data(self):
        return self.data

class EmptyContract:
    def __init__(self, data=None):
        if data != "" and data != b"" and data is not None:
            raise LazyContractValidationError("Empty contract should contain no data")

    def to_dict(self):
        raise TypeError("Cannot convert raw contract to dict, use the to_data() function instead")

    def to_data(self):
        return ""

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

class StreamCommand:
    COMMAND_NAME = "stream"

    # TODO: right now this is a raw string == "true" or "false",
    # we should change this and anyone who uses it to send a boolean property instead
    class Request(RawContract):
        SERIALIZE = False
        data = StringProperty(required=True)

    class Response(RawContract):
        SERIALIZE = False
        data = StringProperty(required=True)

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
        data = StringProperty(required=True)

    class Response(RawContract):
        SERIALIZE = False
        data = StringProperty(required=True)

# Contracts for publishing to streams
class ColorStreamContract(LazyContract):
    STREAM_NAME = "color_mask"
    SERIALIZE = False

    data = BinaryProperty(required=True)
