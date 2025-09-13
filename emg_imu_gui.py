import tkinter as tk
from tkinter import ttk
import random
import time

class EMGIMUDataCollectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EMG/IMU Data Collection - HTN 2025")
        self.root.geometry("700x500")
        self.root.configure(bg='#34495e')
        self.root.minsize(600, 400)
        
        # Center the window on screen
        self.root.eval('tk::PlaceWindow . center')
        
        # Data collection settings
        self.interval = 5000  # 5 seconds in milliseconds
        self.current_action = None
        self.is_collecting = False
        self.collection_timer = None
        self.action_count = 0  # Counter for actions shown
        self.is_resting = False  # Track if we're in a rest period
        
        # Minecraft actions for data collection
        self.actions = [
            "MINING",
            "TURN LEFT", 
            "TURN RIGHT",
            "PLACE BLOCKS"
        ]
        
        # Balanced data collection tracking
        self.action_counts = {action: 0 for action in self.actions}
        self.actions_per_session = 12  # Total actions per session
        self.actions_per_class = 3    # Actions per class (12/4 = 3)
        self.current_session_actions = []  # Track actions in current session
        
        # Action descriptions for better understanding
        self.action_descriptions = {
            "MINING": "Simulate mining blocks (left-click motion)",
            "TURN LEFT": "Turn your body/head to the left",
            "TURN RIGHT": "Turn your body/head to the right", 
            "PLACE BLOCKS": "Simulate placing blocks (right-click motion)"
        }
        
        # Colors
        self.bg_color = '#34495e'
        self.text_color = '#ecf0f1'
        self.action_color = '#e74c3c'  # Red for action prompts
        self.rest_color = '#95a5a6'    # Gray for rest periods
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title label
        title_label = tk.Label(
            main_frame, 
            text="EMG/IMU Data Collection", 
            font=('Arial', 20, 'bold'),
            fg=self.text_color,
            bg=self.bg_color
        )
        title_label.pack(pady=(0, 20))
        
        # Subtitle
        subtitle_label = tk.Label(
            main_frame, 
            text="Minecraft Action Recognition Training", 
            font=('Arial', 14),
            fg='#bdc3c7',
            bg=self.bg_color
        )
        subtitle_label.pack(pady=(0, 25))
        
        # Control buttons frame
        control_frame = tk.Frame(main_frame, bg=self.bg_color)
        control_frame.pack(pady=(0, 20))
        
        # Start/Stop button
        self.control_btn = tk.Button(
            control_frame,
            text="Start Collection",
            font=('Arial', 16, 'bold'),
            bg='#27ae60',  # Green
            fg='#ffffff',  # Pure white text
            activebackground='#229954',
            activeforeground='#ffffff',
            command=self.toggle_collection,
            padx=30,
            pady=10,
            relief='raised',
            bd=4
        )
        self.control_btn.pack(side='left', padx=10)
        
        # Interval setting
        interval_frame = tk.Frame(control_frame, bg=self.bg_color)
        interval_frame.pack(side='left', padx=30)
        
        tk.Label(interval_frame, text="Interval (sec):", 
                font=('Arial', 12, 'bold'), fg=self.text_color, bg=self.bg_color).pack(side='top')
        
        self.interval_var = tk.StringVar(value="5")
        interval_entry = tk.Entry(interval_frame, textvariable=self.interval_var, 
                                width=6, font=('Arial', 14, 'bold'), 
                                bg='#ecf0f1', fg='#2c3e50', justify='center')
        interval_entry.pack(side='top', pady=5)
        interval_entry.bind('<Return>', self.update_interval)
        
        # Action display area (large and prominent)
        self.action_frame = tk.Frame(main_frame, bg=self.bg_color, relief='solid', bd=3)
        self.action_frame.pack(expand=True, fill='both', pady=20)
        
        # Large action text
        self.action_label = tk.Label(
            self.action_frame,
            text="Ready to collect data",
            font=('Arial', 36, 'bold'),
            fg=self.text_color,
            bg=self.bg_color,
            wraplength=600
        )
        self.action_label.pack(expand=True, pady=40)
        
        # Action description
        self.description_label = tk.Label(
            self.action_frame,
            text="Press 'Start Collection' to begin",
            font=('Arial', 14),
            fg='#bdc3c7',
            bg=self.bg_color,
            wraplength=500
        )
        self.description_label.pack(pady=(0, 20))
        
        # Progress and status frame
        status_frame = tk.Frame(main_frame, bg=self.bg_color)
        status_frame.pack(fill='x', pady=(10, 0))
        
        # Progress counter
        self.progress_label = tk.Label(
            status_frame,
            text="Progress: 0/12 actions",
            font=('Arial', 12, 'bold'),
            fg='#3498db',
            bg=self.bg_color
        )
        self.progress_label.pack(side='left')
        
        # Timer display
        self.timer_label = tk.Label(
            status_frame,
            text="Timer: 0.0s",
            font=('Arial', 12, 'bold'),
            fg='#e67e22',
            bg=self.bg_color
        )
        self.timer_label.pack(side='right')
        
        # Class balance display frame
        balance_frame = tk.Frame(main_frame, bg=self.bg_color)
        balance_frame.pack(fill='x', pady=(5, 0))
        
        # Class balance title
        tk.Label(balance_frame, text="Class Balance:", 
                font=('Arial', 10, 'bold'), fg='#95a5a6', bg=self.bg_color).pack()
        
        # Class counters frame
        counters_frame = tk.Frame(balance_frame, bg=self.bg_color)
        counters_frame.pack(pady=5)
        
        # Individual class counters
        self.class_labels = {}
        for action in self.actions:
            label = tk.Label(
                counters_frame,
                text=f"{action}: 0",
                font=('Arial', 9),
                fg='#bdc3c7',
                bg=self.bg_color,
                padx=10
            )
            label.pack(side='left')
            self.class_labels[action] = label
    
    def toggle_collection(self):
        """Start or stop the data collection process"""
        if not self.is_collecting:
            self.start_collection()
        else:
            self.stop_collection()
    
    def start_collection(self):
        """Start the automated action prompting for data collection"""
        self.is_collecting = True
        self.action_count = 0  # Reset counter
        self.is_resting = False
        self.current_session_actions = []  # Reset session tracking
        self.control_btn.config(text="Stop Collection", bg='#e74c3c', 
                               activebackground='#c0392b')
        
        # Generate balanced session plan
        self.generate_balanced_session()
        
        # Start the first action prompt
        self.prompt_next_action()
        
        print(f"EMG/IMU data collection started with {self.interval/1000}s intervals")
        print("Actions: MINING, TURN LEFT, TURN RIGHT, PLACE BLOCKS")
        print("Balanced collection: 3 samples per class")
        print("Rest periods: {:.1f}s between actions, {:.1f}s after 12 actions".format(
            self.interval/2000, self.interval*2/1000))
    
    def stop_collection(self):
        """Stop the data collection process"""
        self.is_collecting = False
        self.is_resting = False
        self.action_count = 0
        self.current_session_actions = []
        self.control_btn.config(text="Start Collection", bg='#27ae60',
                               activebackground='#229954')
        
        # Cancel any pending timer
        if self.collection_timer:
            self.root.after_cancel(self.collection_timer)
            self.collection_timer = None
        
        # Reset display
        self.reset_display()
        
        print("EMG/IMU data collection stopped")
    
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
    
    def generate_balanced_session(self):
        """Generate a balanced sequence of actions for the session"""
        # Create exactly 3 instances of each action
        session_actions = []
        for action in self.actions:
            session_actions.extend([action] * self.actions_per_class)
        
        # Randomize the order
        random.shuffle(session_actions)
        self.current_session_actions = session_actions
        
        print(f"Generated balanced session: {len(session_actions)} actions")
        print(f"Session plan: {session_actions}")
    
    def prompt_next_action(self):
        """Prompt the next action from the balanced session plan"""
        if not self.is_collecting:
            return
        
        # Check if we need a longer rest after 12 actions
        if self.action_count > 0 and self.action_count % 12 == 0:
            self.start_long_rest()
            return
        
        # Check if we have actions left in current session
        if not self.current_session_actions:
            # Generate new balanced session
            self.generate_balanced_session()
        
        # Get next action from the balanced plan
        selected_action = self.current_session_actions.pop(0)
        self.current_action = selected_action
        self.action_count += 1
        
        # Update counters
        self.action_counts[selected_action] += 1
        
        # Update display
        self.action_label.config(text=selected_action, fg=self.action_color)
        self.description_label.config(text=self.action_descriptions[selected_action])
        self.progress_label.config(text=f"Progress: {self.action_count}/12 actions")
        
        # Update class balance display
        self.update_class_balance_display()
        
        # Start timer display
        self.start_time = time.time()
        self.update_timer()
        
        # Log for data collection
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Action prompt: {selected_action} (#{self.action_count}) - Total: {self.action_counts[selected_action]}")
        
        # Schedule rest period after the interval
        self.collection_timer = self.root.after(self.interval, self.start_short_rest)
    
    def update_class_balance_display(self):
        """Update the class balance display"""
        for action in self.actions:
            count = self.action_counts[action]
            self.class_labels[action].config(text=f"{action}: {count}")
    
    def start_short_rest(self):
        """Start short rest period between actions"""
        if not self.is_collecting:
            return
        
        self.is_resting = True
        rest_duration = self.interval // 2  # interval/2 seconds
        
        # Update display for rest
        self.action_label.config(text="REST", fg=self.rest_color)
        self.description_label.config(text=f"Relax for {rest_duration/1000:.1f} seconds")
        
        # Start timer display for rest
        self.start_time = time.time()
        self.rest_duration = rest_duration / 1000
        self.update_timer()
        
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Short rest period ({rest_duration/1000:.1f}s)")
        
        # Schedule next action after rest
        self.collection_timer = self.root.after(rest_duration, self.end_rest)
    
    def start_short_rest(self):
        """Start short rest period between actions"""
        if not self.is_collecting:
            return
        
        self.is_resting = True
        rest_duration = self.interval // 2  # interval/2 seconds
        
        # Update display for rest
        self.action_label.config(text="REST", fg=self.rest_color)
        self.description_label.config(text=f"Relax for {rest_duration/1000:.1f} seconds")
        
        # Start timer display for rest
        self.start_time = time.time()
        self.rest_duration = rest_duration / 1000
        self.update_timer()
        
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Short rest period ({rest_duration/1000:.1f}s)")
        
        # Schedule next action after rest
        self.collection_timer = self.root.after(rest_duration, self.end_rest)
    
    def start_long_rest(self):
        """Start long rest period after 12 actions"""
        if not self.is_collecting:
            return
        
        self.is_resting = True
        rest_duration = self.interval * 2  # interval*2 seconds
        
        # Update display for long rest
        self.action_label.config(text="LONG REST", fg=self.rest_color)
        self.description_label.config(text=f"Extended break for {rest_duration/1000:.1f} seconds - Completed {self.action_count} actions")
        
        # Start timer display for long rest
        self.start_time = time.time()
        self.rest_duration = rest_duration / 1000
        self.update_timer()
        
        timestamp = time.time()
        print(f"[{timestamp:.3f}] Long rest period ({rest_duration/1000:.1f}s) after {self.action_count} actions")
        
        # Schedule next action after long rest
        self.collection_timer = self.root.after(rest_duration, self.end_rest)
    
    def end_rest(self):
        """End rest period and continue with next action"""
        if not self.is_collecting:
            return
        
        self.is_resting = False
        self.prompt_next_action()  # Use balanced action prompting
    
    def update_timer(self):
        """Update the timer display"""
        if not self.is_collecting:
            self.timer_label.config(text="Timer: 0.0s")
            return
        
        elapsed = time.time() - self.start_time
        
        if self.is_resting:
            remaining = max(0, self.rest_duration - elapsed)
            self.timer_label.config(text=f"Rest: {remaining:.1f}s")
        else:
            max_time = self.interval / 1000
            remaining = max(0, max_time - elapsed)
            self.timer_label.config(text=f"Action: {remaining:.1f}s")
        
        # Continue updating timer
        if self.is_collecting:
            self.root.after(100, self.update_timer)  # Update every 100ms
    
    def reset_display(self):
        """Reset the display to initial state"""
        self.action_label.config(text="Ready to collect data", fg=self.text_color)
        self.description_label.config(text="Press 'Start Collection' to begin")
        self.progress_label.config(text="Progress: 0/12 actions")
        self.timer_label.config(text="Timer: 0.0s")
        self.current_action = None
        
        # Reset class counters display
        for action in self.actions:
            self.class_labels[action].config(text=f"{action}: 0")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = EMGIMUDataCollectionGUI()
    app.run()