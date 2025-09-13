import tkinter as tk
from tkinter import ttk
import random
import time
import EEGCollector


class ArrowGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BCI Data Collection - HTN 2025")
        self.root.geometry("1920x1080")
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
        
        # Balanced data collection tracking
        self.directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.direction_counts = {direction: 0 for direction in self.directions}
        self.directions_per_session = 12  # Total directions per session
        self.directions_per_class = 3     # Directions per class (12/4 = 3)
        self.current_session_directions = []  # Track directions in current session
        self.session_number = 0  # Track session count
        self.total_sets = 1  # Number of sets to collect
        self.current_set = 0  # Current set being collected
        
        # Colors
        self.normal_color = '#ecf0f1'  # White/light gray
        self.highlight_color = '#e67e22'  # Orange

        self.eeg = EEGCollector.EEGCollector(debug=False)
        
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
        title_label.pack(pady=(0, 2))
        
        # Control buttons frame
        control_frame = tk.Frame(main_frame, bg='#2c3e50')
        control_frame.pack(pady=(0, 2))
        
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
        interval_frame.pack(side='left', padx=2)
        
        tk.Label(interval_frame, text="Interval (sec):", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2c3e50').pack(side='left')
        
        self.interval_var = tk.StringVar(value="5")
        interval_entry = tk.Entry(interval_frame, textvariable=self.interval_var, 
                                width=5, font=('Arial', 11, 'bold'), 
                                bg='#ecf0f1', fg='#2c3e50')
        interval_entry.pack(side='left', padx=(5, 0))
        interval_entry.bind('<Return>', self.update_interval)
        
        # Number of sets setting
        sets_frame = tk.Frame(control_frame, bg='#2c3e50')
        sets_frame.pack(side='left', padx=20)
        
        tk.Label(sets_frame, text="Sets:", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2c3e50').pack(side='top')
        
        self.sets_var = tk.StringVar(value="1")
        sets_entry = tk.Entry(sets_frame, textvariable=self.sets_var, 
                            width=5, font=('Arial', 11, 'bold'), 
                            bg='#ecf0f1', fg='#2c3e50', justify='center')
        sets_entry.pack(side='top', pady=2)
        sets_entry.bind('<Return>', self.update_sets)
        
        # Status label to show current state
        self.status_label = tk.Label(
            main_frame,
            text="Press 'Start Collection' to begin data collection",
            font=('Arial', 12),
            fg='#3498db',
            bg='#2c3e50'
        )
        self.status_label.pack(pady=(0, 2))
        
        # Session display frame
        session_frame = tk.Frame(main_frame, bg='#2c3e50', relief='sunken', bd=2)
        session_frame.pack(fill='x', padx=20, pady=(0, 2))
        
        # Session title
        session_title = tk.Label(
            session_frame,
            text="Current Session (12 directions):",
            font=('Arial', 10, 'bold'),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        session_title.pack(pady=(1, 0))
        
        # Session directions display (text box)
        self.session_text = tk.Text(
            session_frame,
            height=2,
            width=25,
            font=('Arial', 9),
            bg='#34495e',
            fg='#ecf0f1',
            relief='flat',
            bd=0,
            wrap='word',
            state='disabled'  # Read-only
        )
        self.session_text.pack(padx=10, pady=(1, 1))
        
        # Class balance display frame
        balance_frame = tk.Frame(main_frame, bg='#2c3e50')
        balance_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        # Class balance title
        tk.Label(balance_frame, text="Class Balance:", 
                font=('Arial', 10, 'bold'), fg='#95a5a6', bg='#2c3e50').pack()
        
        # Class counters frame
        counters_frame = tk.Frame(balance_frame, bg='#2c3e50')
        counters_frame.pack(pady=5)
        
        # Individual class counters
        self.class_labels = {}
        for direction in self.directions:
            label = tk.Label(
                counters_frame,
                text=f"{direction}: 0",
                font=('Arial', 9),
                fg='#bdc3c7',
                bg='#2c3e50',
                padx=10
            )
            label.pack(side='left')
            self.class_labels[direction] = label
        
        # Arrow symbols container
        self.arrow_frame = tk.Frame(main_frame, bg='#2c3e50')
        self.arrow_frame.pack(expand=True)
        
        # Initial font size (will be scaled)
        self.base_font_size = 1
        
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
        self.up_label.grid(row=0, column=1, padx=0, pady=0)
        self.up_label.bind("<Button-1>", lambda e: self.arrow_clicked("UP"))
        self.arrow_labels["UP"] = self.up_label
        
        # Left and Right arrows (middle row)
        self.left_label = tk.Label(
            self.arrow_frame,
            text="◀",  # Solid triangle arrow for consistency
            **arrow_style
        )
        self.left_label.grid(row=1, column=0, padx=0, pady=0)
        self.left_label.bind("<Button-1>", lambda e: self.arrow_clicked("LEFT"))
        self.arrow_labels["LEFT"] = self.left_label
        
        self.right_label = tk.Label(
            self.arrow_frame,
            text="▶",  # Solid triangle arrow for consistency
            **arrow_style
        )
        self.right_label.grid(row=1, column=2, padx=0, pady=0)
        self.right_label.bind("<Button-1>", lambda e: self.arrow_clicked("RIGHT"))
        self.arrow_labels["RIGHT"] = self.right_label
        
        # Down arrow (bottom row)
        self.down_label = tk.Label(
            self.arrow_frame,
            text="▼",  # Solid triangle arrow for consistency
            **arrow_style
        )
        self.down_label.grid(row=2, column=1, padx=0, pady=(0, 500))
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
        self.current_session_directions = []  # Reset session tracking
        self.current_set = 1  # Start with set 1
        self.update_sets()  # Update total_sets from input field
        
        self.control_btn.config(text="Stop Collection", bg='#e74c3c', 
                               activebackground='#c0392b')
        self.status_label.config(text=f"Data collection started - Set 1/{self.total_sets}")
        
        # Generate balanced session plan
        self.generate_balanced_session()
        
        # Start the first highlight cycle
        self.start_short_rest()
        
        print(f"Data collection started with {self.interval/1000}s intervals")
        print(f"Collecting {self.total_sets} set(s) of balanced data")
        print("Balanced collection: 3 samples per direction")
        print("Rest periods: {:.1f}s between directions, {:.1f}s after 12 directions".format(
            self.interval/2000, self.interval*2/1000))
    
    def stop_collection(self):
        """Stop the data collection process"""
        self.is_collecting = False
        self.is_resting = False
        self.direction_count = 0
        self.current_session_directions = []
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
    
    def update_sets(self, event=None):
        """Update the number of sets from user input"""
        try:
            new_sets = int(self.sets_var.get())
            if new_sets > 0:
                self.total_sets = new_sets
                print(f"Number of sets updated to {new_sets}")
            else:
                self.sets_var.set("1")  # Reset to default
        except ValueError:
            self.sets_var.set("1")  # Reset to default if invalid
    
    def generate_balanced_session(self):
        """Generate a balanced sequence of directions for the session"""
        # Create exactly 3 instances of each direction
        session_directions = []
        for direction in self.directions:
            session_directions.extend([direction] * self.directions_per_class)
        
        # Randomize the order
        random.shuffle(session_directions)
        self.current_session_directions = session_directions
        self.session_number += 1
        
        # Update session display
        self.update_session_display()
        
        print(f"Generated balanced session #{self.session_number}: {len(session_directions)} directions")
        print(f"Session plan: {session_directions}")
    
    def update_session_display(self):
        """Update the session directions display text box"""
        self.session_text.config(state='normal')
        self.session_text.delete(1.0, tk.END)
        
        if self.current_session_directions:
            # Format directions in a readable way
            directions_str = " → ".join(self.current_session_directions)
            self.session_text.insert(1.0, f"Session #{self.session_number}: {directions_str}")
        else:
            self.session_text.insert(1.0, "No session planned")
        
        self.session_text.config(state='disabled')
    
    def update_class_balance_display(self):
        """Update the class balance display"""
        for direction in self.directions:
            count = self.direction_counts[direction]
            self.class_labels[direction].config(text=f"{direction}: {count}")
    
    def highlight_next_direction(self):
        """Highlight the next direction from the balanced session plan"""
        if not self.is_collecting:
            return
        
        # Check if we need a longer rest after 12 directions
        if self.direction_count > 0 and self.direction_count % 12 == 0:
            self.start_long_rest()
            return
        
        # Check if we have directions left in current session
        if not self.current_session_directions:
            # Generate new balanced session
            self.generate_balanced_session()
        
        # Get next direction from the balanced plan
        selected_direction = self.current_session_directions.pop(0)
        self.current_highlighted = selected_direction
        self.direction_count += 1
        
        # Update counters
        self.direction_counts[selected_direction] += 1
        
        # Reset all arrows to normal color first
        self.reset_all_arrows()
        
        # Highlight the selected arrow
        selected_label = self.arrow_labels[selected_direction]
        selected_label.config(fg=self.highlight_color)  # Orange color
        self.eeg.setMarker(selected_direction)
        
        # Update status
        self.status_label.config(text=f"Set {self.current_set}/{self.total_sets} - Focus on: {selected_direction} ({self.direction_count}/12)")
        
        # Update displays
        self.update_class_balance_display()
        self.update_session_display()
        
        # Log for data collection
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Highlighted: {selected_direction} (#{self.direction_count}) - Total: {self.direction_counts[selected_direction]}")
        
        # Schedule rest period (interval), then short rest
        duration = self.interval+random.randint(-1000, 1000)
        self.collection_timer = self.root.after(duration, self.start_short_rest)

        print(f"[{time.time():.3f}] Shown Stimulus for {duration:.3f} seconds")
    
    def start_short_rest(self):
        """Start short rest period between directions"""
        if not self.is_collecting:
            return
        
        self.eeg.setMarker("stop")
        self.is_resting = True
        self.reset_all_arrows()
        
        rest_duration = self.interval - 1 + random.randint(-1000, 1000)
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
        
        # Check if we've completed the current set
        if self.current_set >= self.total_sets:
            # Completed all sets
            self.status_label.config(text=f"Collection Complete! Finished {self.total_sets} set(s)")
            print(f"Data collection completed! Collected {self.total_sets} set(s)")
            self.stop_collection()
            return
        
        # Move to next set
        self.current_set += 1
        self.direction_count = 0  # Reset direction counter for new set
        
        rest_duration = self.interval * 3  # interval*2 seconds
        self.status_label.config(text=f"Set Break... ({rest_duration/1000:.1f}s) - Starting Set {self.current_set}/{self.total_sets}")
        
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Set break ({rest_duration/1000:.1f}s) - Starting set {self.current_set}/{self.total_sets}")
        
        # Schedule next highlight after long rest
        self.collection_timer = self.root.after(rest_duration, self.end_rest)
    
    def end_rest(self):
        """End rest period and continue with next highlight"""
        if not self.is_collecting:
            return
        
        self.is_resting = False
        self.highlight_next_direction()  # Use balanced highlighting
    
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