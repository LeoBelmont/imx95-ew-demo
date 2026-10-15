import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

VIDEO_PATH = "input.mp4"

def main():
    Gst.init(None)

    # Same as your terminal GStreamer command
    pipeline_str = (
        f'filesrc location={VIDEO_PATH} ! '
        f'qtdemux ! h264parse ! v4l2h264dec ! '
        f'videoconvert ! waylandsink sync=false'
    )

    pipeline = Gst.parse_launch(pipeline_str)
    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
