# Key remapper: makes 'i' behave like an additional 'w' key for Minecraft (or any app)
# macOS note: Grant your terminal/IDE Accessibility permission (System Settings > Privacy & Security > Accessibility)
# Stop with Ctrl+C.

from pynput import keyboard
from pynput.keyboard import Controller

controller = Controller()

state = {
    'i_down': False,       # Is the i key currently held
    'w_physical': False,   # Is the physical w key held
    'w_virtual': False     # Are we currently holding a virtual w (pressed by script)
}

def press_w_virtual():
    if not state['w_virtual'] and not state['w_physical']:
        controller.press('w')
        state['w_virtual'] = True


def release_w_virtual():
    if state['w_virtual'] and not state['w_physical']:
        controller.release('w')
        state['w_virtual'] = False


def on_press(key):
    try:
        ch = key.char
    except AttributeError:
        return

    if ch == 'i':
        if not state['i_down']:
            state['i_down'] = True
            press_w_virtual()
    elif ch == 'w':
        if not state['w_physical']:
            state['w_physical'] = True
            # If we had a virtual w, convert responsibility to physical key
            if state['w_virtual']:
                state['w_virtual'] = False  # Avoid double release later


def on_release(key):
    try:
        ch = key.char
    except AttributeError:
        return

    if ch == 'i':
        state['i_down'] = False
        release_w_virtual()
    elif ch == 'w':
        state['w_physical'] = False
        if state['i_down']:
            # Still want forward movement because i is held
            press_w_virtual()
        else:
            release_w_virtual()


def main():
    print("Key mapper running: 'i' acts like 'w'. Ctrl+C to exit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    main()
