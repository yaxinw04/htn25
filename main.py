import tkinter as tk
from tkinter import ttk
import random
import time

class ArrowGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BCI Data Collection - HTN 2025")
        self.root.geometry("500x400")
        self.root.configure(bg='#2c3e50')
        self.root.minsize(300, 250)  # Set minimum window size
        
        # Center the window on screen
        self.root.eval('tk::PlaceWindow . center')
        
        # Store arrow labels for scaling
        self.arrow_labels = {}
        
        # Data collection settings
        self.interval = 5000  # 5 seconds in milliseconds
        self.current_highlighted = None
        self.is_collecting = False
        self.collection_timer = None
        self.direction_count = 0  # Counter for directions shown
        self.is_resting = False  # Track if we're in a rest period
        
        # Colors
        self.normal_color = '#ecf0f1'  # White/light gray
        self.highlight_color = '#e67e22'  # Orange
        
        self.setup_ui()
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
    
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title label
        title_label = tk.Label(
            main_frame, 
            text="BCI Data Collection", 
            font=('Arial', 18, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack(pady=(0, 15))
        
        # Control buttons frame
        control_frame = tk.Frame(main_frame, bg='#2c3e50')
        control_frame.pack(pady=(0, 15))
        
        # Start/Stop button - improved visibility
        self.control_btn = tk.Button(
            control_frame,
            text="Start Collection",
            font=('Arial', 14, 'bold'),
            bg='#2ecc71',  # Brighter green
            fg='#ffffff',  # Pure white text
            activebackground='#27ae60',
            activeforeground='#ffffff',
            command=self.toggle_collection,
            padx=25,
            pady=8,
            relief='raised',
            bd=3
        )
        self.control_btn.pack(side='left', padx=5)
        
        # Interval setting
        interval_frame = tk.Frame(control_frame, bg='#2c3e50')
        interval_frame.pack(side='left', padx=20)
        
        tk.Label(interval_frame, text="Interval (sec):", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2c3e50').pack(side='left')
        
        self.interval_var = tk.StringVar(value="5")
        interval_entry = tk.Entry(interval_frame, textvariable=self.interval_var, 
                                width=5, font=('Arial', 11, 'bold'), 
                                bg='#ecf0f1', fg='#2c3e50')
        interval_entry.pack(side='left', padx=(5, 0))
        interval_entry.bind('<Return>', self.update_interval)
        
        # Status label to show current state
        self.status_label = tk.Label(
            main_frame,
            text="Press 'Start Collection' to begin data collection",
            font=('Arial', 12),
            fg='#3498db',
            bg='#2c3e50'
        )
        self.status_label.pack(pady=(0, 20))
        
        # Arrow symbols container
        self.arrow_frame = tk.Frame(main_frame, bg='#2c3e50')
        self.arrow_frame.pack(expand=True)
        
        # Initial font size (will be scaled)
        self.base_font_size = 48

        padx, pady = 80, 60  # Even more padding for maximum spacing from center
        
        # Configure arrow symbol style with thicker arrows
        arrow_style = {
            'font': ('Arial Black', self.base_font_size, 'bold'),
            'fg': self.normal_color,  # Use normal color initially
            'bg': '#2c3e50',
            'cursor': 'hand2'
        }
        
        # Create arrow symbols in cross pattern using consistent Unicode arrows
        # Up arrow (top row)
        self.up_label = tk.Label(
            self.arrow_frame,
            text="▲",  # Solid triangle arrow for consistency
            **arrow_style
        )
        self.up_label.grid(row=0, column=1, padx=padx, pady=pady)
        self.up_label.bind("<Button-1>", lambda e: self.arrow_clicked("UP"))
        self.arrow_labels["UP"] = self.up_label
        
        # Left and Right arrows (middle row)
        self.left_label = tk.Label(
            self.arrow_frame,
            text="◀",  # Solid triangle arrow for consistency
            **arrow_style
        )
        self.left_label.grid(row=1, column=0, padx=padx, pady=pady)
        self.left_label.bind("<Button-1>", lambda e: self.arrow_clicked("LEFT"))
        self.arrow_labels["LEFT"] = self.left_label
        
        self.right_label = tk.Label(
            self.arrow_frame,
            text="▶",  # Solid triangle arrow for consistency
            **arrow_style
        )
        self.right_label.grid(row=1, column=2, padx=padx, pady=pady)
        self.right_label.bind("<Button-1>", lambda e: self.arrow_clicked("RIGHT"))
        self.arrow_labels["RIGHT"] = self.right_label
        
        # Down arrow (bottom row)
        self.down_label = tk.Label(
            self.arrow_frame,
            text="▼",  # Solid triangle arrow for consistency
            **arrow_style
        )
        self.down_label.grid(row=2, column=1, padx=padx, pady=pady)
        self.down_label.bind("<Button-1>", lambda e: self.arrow_clicked("DOWN"))
        self.arrow_labels["DOWN"] = self.down_label
        
        # Keyboard bindings
        self.root.bind('<Up>', lambda e: self.arrow_clicked("UP"))
        self.root.bind('<Down>', lambda e: self.arrow_clicked("DOWN"))
        self.root.bind('<Left>', lambda e: self.arrow_clicked("LEFT"))
        self.root.bind('<Right>', lambda e: self.arrow_clicked("RIGHT"))
        
        # Make window focusable for keyboard events
        self.root.focus_set()
        
    def arrow_clicked(self, direction):
        """Handle arrow symbol clicks"""
        self.status_label.config(text=f"You pressed: {direction}")
        
        # Brief visual feedback - change color of the arrow symbol
        label_map = {
            "UP": self.up_label,
            "DOWN": self.down_label,
            "LEFT": self.left_label,
            "RIGHT": self.right_label
        }
        
        label = label_map[direction]
        original_color = label.cget('fg')
        label.config(fg='#e74c3c')  # Flash red
        self.root.after(200, lambda: label.config(fg=original_color))
        
        print(f"Arrow pressed: {direction}")  # Console output for debugging
    
    def on_window_resize(self, event):
        """Handle window resize events to scale arrows"""
        # Only respond to root window resize events
        if event.widget == self.root:
            # Calculate new font size based on window dimensions
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            # Use the smaller dimension to maintain aspect ratio
            min_dimension = min(window_width, window_height)
            
            # Scale font size (minimum 20, maximum 120)
            new_font_size = max(20, min(120, int(min_dimension * 0.12)))
            
            # Update all arrow labels with new font size
            for label in self.arrow_labels.values():
                current_font = label.cget('font')
                # Parse current font and update size
                if isinstance(current_font, tuple):
                    font_family, _, font_weight = current_font
                else:
                    # Handle string font format
                    font_family = 'Arial Black'
                    font_weight = 'bold'
                
                label.config(font=(font_family, new_font_size, font_weight))
    
    def toggle_collection(self):
        """Start or stop the data collection process"""
        if not self.is_collecting:
            self.start_collection()
        else:
            self.stop_collection()
    
    def start_collection(self):
        """Start the automated arrow highlighting for data collection"""
        self.is_collecting = True
        self.direction_count = 0  # Reset counter
        self.is_resting = False
        self.control_btn.config(text="Stop Collection", bg='#e74c3c', 
                               activebackground='#c0392b')
        self.status_label.config(text="Data collection started - Focus on the highlighted arrow!")
        
        # Start the first highlight cycle
        self.highlight_random_arrow()
        
        print(f"Data collection started with {self.interval/1000}s intervals")
        print("Rest periods: {:.1f}s between directions, {:.1f}s after 12 directions".format(
            self.interval/2000, self.interval*2/1000))
    
    def stop_collection(self):
        """Stop the data collection process"""
        self.is_collecting = False
        self.is_resting = False
        self.direction_count = 0
        self.control_btn.config(text="Start Collection", bg='#2ecc71',
                               activebackground='#27ae60')
        self.status_label.config(text="Data collection stopped")
        
        # Cancel any pending timer
        if self.collection_timer:
            self.root.after_cancel(self.collection_timer)
            self.collection_timer = None
        
        # Reset all arrows to normal color
        self.reset_all_arrows()
        
        print("Data collection stopped")
    
    def update_interval(self, event=None):
        """Update the collection interval from user input"""
        try:
            new_interval = float(self.interval_var.get())
            if new_interval > 0:
                self.interval = int(new_interval * 1000)  # Convert to milliseconds
                print(f"Interval updated to {new_interval} seconds")
            else:
                self.interval_var.set("5")  # Reset to default
        except ValueError:
            self.interval_var.set("5")  # Reset to default if invalid
    
    def highlight_random_arrow(self):
        """Randomly select and highlight an arrow with rest periods"""
        if not self.is_collecting:
            return
        
        # Check if we need a longer rest after 12 directions
        if self.direction_count > 0 and self.direction_count % 12 == 0:
            self.start_long_rest()
            return
        
        # Reset previous highlight
        self.reset_all_arrows()
        
        # Choose random direction
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        selected_direction = random.choice(directions)
        self.current_highlighted = selected_direction
        self.direction_count += 1
        
        # Highlight the selected arrow
        selected_label = self.arrow_labels[selected_direction]
        selected_label.config(fg=self.highlight_color)  # Orange color
        
        # Update status
        self.status_label.config(text=f"Focus on: {selected_direction} ({self.direction_count}/12)")
        
        # Log for data collection
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Highlighted: {selected_direction} (#{self.direction_count})")
        
        # Schedule rest period (interval/2), then next highlight
        self.collection_timer = self.root.after(self.interval, self.start_short_rest)
    
    def start_short_rest(self):
        """Start short rest period between directions"""
        if not self.is_collecting:
            return
        
        self.is_resting = True
        self.reset_all_arrows()
        
        rest_duration = self.interval // 2  # interval/2 seconds
        self.status_label.config(text=f"Rest period... ({rest_duration/1000:.1f}s)")
        
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Short rest period ({rest_duration/1000:.1f}s)")
        
        # Schedule next highlight after rest
        self.collection_timer = self.root.after(rest_duration, self.end_rest)
    
    def start_long_rest(self):
        """Start long rest period after 12 directions"""
        if not self.is_collecting:
            return
        
        self.is_resting = True
        self.reset_all_arrows()
        
        rest_duration = self.interval * 2  # interval*2 seconds
        self.status_label.config(text=f"Long rest period... ({rest_duration/1000:.1f}s) - Completed {self.direction_count} directions")
        
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Long rest period ({rest_duration/1000:.1f}s) after {self.direction_count} directions")
        
        # Schedule next highlight after long rest
        self.collection_timer = self.root.after(rest_duration, self.end_rest)
    
    def end_rest(self):
        """End rest period and continue with next highlight"""
        if not self.is_collecting:
            return
        
        self.is_resting = False
        self.highlight_random_arrow()
    
    def reset_all_arrows(self):
        """Reset all arrows to normal color"""
        for label in self.arrow_labels.values():
            label.config(fg=self.normal_color)
        self.current_highlighted = None
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ArrowGUI()
    app.run()