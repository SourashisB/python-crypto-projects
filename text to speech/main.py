import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pyttsx3
import threading
import requests
import nltk
from nltk.corpus import wordnet

# Download required NLTK data (only needed first time)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class DictionaryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaking Dictionary")
        self.root.geometry("600x500")
        self.root.minsize(500, 400)
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12))
        self.style.configure("TLabel", font=("Helvetica", 12))
        self.style.configure("Header.TLabel", font=("Helvetica", 18, "bold"))
        
        # Set color scheme
        self.bg_color = "#f0f4f8"
        self.accent_color = "#4a6fa5"
        self.text_color = "#333333"
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_label = ttk.Label(main_frame, text="Speaking Dictionary", 
                                 style="Header.TLabel")
        header_label.pack(pady=(0, 20))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=10)
        
        # Word label and entry
        word_label = ttk.Label(input_frame, text="Enter a word:")
        word_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.word_var = tk.StringVar()
        self.word_entry = ttk.Entry(input_frame, textvariable=self.word_var, 
                                   font=("Helvetica", 12), width=25)
        self.word_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        self.word_entry.focus()
        
        # Look up button
        search_button = ttk.Button(input_frame, text="Look Up", 
                                  command=self.lookup_word)
        search_button.pack(side=tk.RIGHT)
        
        # Bind Enter key to lookup_word function
        self.word_entry.bind("<Return>", lambda event: self.lookup_word())
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Definition", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Definition display area
        self.definition_text = scrolledtext.ScrolledText(
            result_frame, wrap=tk.WORD, font=("Helvetica", 11),
            bg="white", fg=self.text_color, height=10
        )
        self.definition_text.pack(fill=tk.BOTH, expand=True)
        self.definition_text.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Source selection
        self.source_var = tk.StringVar(value="both")
        source_frame = ttk.LabelFrame(button_frame, text="Dictionary Source")
        source_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(source_frame, text="WordNet", variable=self.source_var, 
                       value="wordnet").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Dictionary API", variable=self.source_var, 
                       value="api").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Both", variable=self.source_var, 
                       value="both").pack(side=tk.LEFT, padx=5)
        
        # Speak button
        self.speak_button = ttk.Button(button_frame, text="Speak Definition",
                                     command=self.speak_definition)
        self.speak_button.pack(side=tk.LEFT, padx=(10, 10))
        self.speak_button.state(['disabled'])
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="Stop Speaking",
                                    command=self.stop_speaking)
        self.stop_button.pack(side=tk.LEFT)
        self.stop_button.state(['disabled'])
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="Clear",
                                command=self.clear_all)
        clear_button.pack(side=tk.RIGHT)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Store current definition
        self.current_definition = ""
        
    def lookup_word(self):
        word = self.word_var.get().strip().lower()
        if not word:
            messagebox.showwarning("Empty Input", "Please enter a word to look up.")
            return
            
        self.status_var.set(f"Looking up '{word}'...")
        self.root.update_idletasks()
        
        # Use a thread to prevent GUI freezing
        threading.Thread(target=self._fetch_definition, args=(word,), daemon=True).start()
    
    def _fetch_definition(self, word):
        try:
            source = self.source_var.get()
            wordnet_result = None
            api_result = None
            
            # Get definitions based on selected source
            if source in ["wordnet", "both"]:
                wordnet_result = self._get_wordnet_definition(word)
                
            if source in ["api", "both"]:
                api_result = self._get_api_definition(word)
            
            # Format and display results
            self.definition_text.config(state=tk.NORMAL)
            self.definition_text.delete(1.0, tk.END)
            
            formatted_text = f"Definitions for '{word}':\n\n"
            found_definitions = False
            
            if wordnet_result:
                formatted_text += "--- WordNet Dictionary ---\n"
                for pos, definitions in wordnet_result.items():
                    formatted_text += f"{pos}:\n"
                    for i, definition in enumerate(definitions, 1):
                        formatted_text += f"  {i}. {definition}\n"
                    formatted_text += "\n"
                found_definitions = True
            
            if api_result:
                if wordnet_result:
                    formatted_text += "\n--- Dictionary API ---\n"
                formatted_text += api_result
                found_definitions = True
            
            if found_definitions:
                self.definition_text.insert(tk.END, formatted_text)
                self.current_definition = formatted_text
                self.speak_button.state(['!disabled'])
                self.status_var.set(f"Found definition for '{word}'")
            else:
                self.definition_text.insert(tk.END, f"No definition found for '{word}'")
                self.current_definition = ""
                self.speak_button.state(['disabled'])
                self.status_var.set(f"No definition found for '{word}'")
                
            self.definition_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.definition_text.config(state=tk.NORMAL)
            self.definition_text.delete(1.0, tk.END)
            self.definition_text.insert(tk.END, f"Error: {str(e)}")
            self.definition_text.config(state=tk.DISABLED)
            self.current_definition = ""
            self.speak_button.state(['disabled'])
            self.status_var.set("Error occurred during lookup")
    
    def _get_wordnet_definition(self, word):
        """Get definitions from WordNet"""
        synsets = wordnet.synsets(word)
        if not synsets:
            return None
            
        result = {}
        pos_mapping = {
            'n': 'Noun',
            'v': 'Verb',
            'a': 'Adjective',
            's': 'Adjective Satellite',
            'r': 'Adverb'
        }
        
        for synset in synsets:
            pos = pos_mapping.get(synset.pos(), synset.pos())
            if pos not in result:
                result[pos] = []
            if synset.definition() not in result[pos]:
                result[pos].append(synset.definition())
                
        return result
    
    def _get_api_definition(self, word):
        """Get definition from Free Dictionary API"""
        try:
            response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    formatted_text = ""
                    
                    for entry in data:
                        if 'meanings' in entry:
                            for meaning in entry['meanings']:
                                pos = meaning.get('partOfSpeech', 'Unknown')
                                formatted_text += f"{pos.capitalize()}:\n"
                                
                                if 'definitions' in meaning:
                                    for i, def_item in enumerate(meaning['definitions'], 1):
                                        definition = def_item.get('definition', '')
                                        if definition:
                                            formatted_text += f"  {i}. {definition}\n"
                                            
                                            # Add example if available
                                            example = def_item.get('example', '')
                                            if example:
                                                formatted_text += f"     Example: {example}\n"
                                                
                                formatted_text += "\n"
                                
                            # Add synonyms if available
                            if 'synonyms' in meaning and meaning['synonyms']:
                                formatted_text += f"  Synonyms: {', '.join(meaning['synonyms'][:5])}\n\n"
                    
                    # Add phonetics if available
                    if 'phonetics' in entry and entry['phonetics']:
                        for phonetic in entry['phonetics']:
                            if 'text' in phonetic and phonetic['text']:
                                formatted_text += f"Pronunciation: {phonetic['text']}\n\n"
                                break
                    
                    return formatted_text
            return None
        except Exception as e:
            print(f"API error: {str(e)}")
            return None
    
    def speak_definition(self):
        if not self.current_definition:
            messagebox.showinfo("No Definition", "No definition available to speak.")
            return
            
        self.speak_button.state(['disabled'])
        self.stop_button.state(['!disabled'])
        self.status_var.set("Speaking definition...")
        
        # Use a thread for speech to prevent GUI freezing
        self.speak_thread = threading.Thread(target=self._speak_text, 
                                            args=(self.current_definition,), 
                                            daemon=True)
        self.speak_thread.start()
    
    def _speak_text(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {str(e)}")
        finally:
            # Re-enable buttons in the main thread
            self.root.after(0, self._reset_after_speech)
    
    def _reset_after_speech(self):
        self.speak_button.state(['!disabled'])
        self.stop_button.state(['disabled'])
        self.status_var.set("Ready")
    
    def stop_speaking(self):
        self.engine.stop()
        self.stop_button.state(['disabled'])
        self.speak_button.state(['!disabled'])
        self.status_var.set("Speech stopped")
    
    def clear_all(self):
        self.word_var.set("")
        self.definition_text.config(state=tk.NORMAL)
        self.definition_text.delete(1.0, tk.END)
        self.definition_text.config(state=tk.DISABLED)
        self.current_definition = ""
        self.speak_button.state(['disabled'])
        self.stop_button.state(['disabled'])
        self.status_var.set("Ready")
        self.word_entry.focus()

def main():
    root = tk.Tk()
    app = DictionaryApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()