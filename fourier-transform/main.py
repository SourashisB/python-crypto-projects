import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from mpl_toolkits.mplot3d import Axes3D

# ------------------- Fourier Transform Function -------------------

def custom_dft2(f):
    """Compute the 2D Discrete Fourier Transform manually."""
    M, N = f.shape
    F = np.zeros((M, N), dtype=complex)
    for u in range(M):
        for v in range(N):
            sum_val = 0.0
            for x in range(M):
                for y in range(N):
                    angle = -2j * np.pi * ((u * x) / M + (v * y) / N)
                    sum_val += f[x, y] * np.exp(angle)
            F[u, v] = sum_val
    return F

# ------------------- GUI and Plotting -------------------

class FourierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Transform Visualizer")

        self.expression = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        entry = tk.Entry(self.root, textvariable=self.expression, font=("Arial", 16), width=40)
        entry.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        buttons = [
            'sin', 'cos', 'pi', '*', '/',
            'x', 'y', '+', '-', '(',
            ')', '0.1', '2', 'np.', 'Clear',
            'Plot'
        ]

        row = 1
        col = 0
        for btn in buttons:
            action = lambda x=btn: self.on_button_click(x)
            tk.Button(self.root, text=btn, width=8, height=2, command=action).grid(row=row, column=col)
            col += 1
            if col > 4:
                col = 0
                row += 1

    def on_button_click(self, char):
        if char == "Clear":
            self.expression.set("")
        elif char == "Plot":
            self.plot_wave()
        else:
            current = self.expression.get()
            self.expression.set(current + char)

    def plot_wave(self):
        try:
            # Define spatial domain
            x = np.linspace(-10, 10, 50)
            y = np.linspace(-10, 10, 50)
            X, Y = np.meshgrid(x, y)

            # Evaluate the user-defined wave equation
            expr = self.expression.get()
            local_dict = {"x": X, "y": Y, "np": np, "sin": np.sin, "cos": np.cos, "pi": np.pi}
            wave = eval(expr, {"__builtins__": {}}, local_dict)

            # Compute custom DFT
            fourier_transform = custom_dft2(wave)
            fourier_shifted = np.fft.fftshift(fourier_transform)
            magnitude_spectrum = np.abs(fourier_shifted)

            # Create frequency axes
            kx = np.fft.fftshift(np.fft.fftfreq(x.size, d=(x[1] - x[0])))
            ky = np.fft.fftshift(np.fft.fftfreq(y.size, d=(y[1] - y[0])))
            KX, KY = np.meshgrid(kx, ky)

            # Plotting
            fig = plt.figure(figsize=(14, 6))

            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.plot_surface(X, Y, wave, cmap='viridis')
            ax1.set_title('User Defined Wave')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Amplitude')

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.plot_surface(KX, KY, magnitude_spectrum, cmap='plasma')
            ax2.set_title('Custom DFT Magnitude Spectrum')
            ax2.set_xlabel('Frequency X')
            ax2.set_ylabel('Frequency Y')
            ax2.set_zlabel('|F(kx, ky)|')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to evaluate expression:\n{e}")

# ------------------- Run Application -------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = FourierApp(root)
    root.mainloop()