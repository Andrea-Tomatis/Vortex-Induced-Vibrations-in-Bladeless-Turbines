from dataclasses import dataclass

# Configuration Data Structure
@dataclass
class SimConfig:
    nx: int = 600
    ny: int = 200
    Re: float = 250.0
    uLB: float = 0.02
    max_iter: int = 5000
    L_char: float = 20.0  
    
    # Environment boundaries
    top_bottom_walls: bool = False  # True = Closed Wind Tunnel, False = Open Space

    flow_profile: str = 'uniform'  # Can be 'uniform' or 'shear'
    
    save_video: bool = True
    video_filename: str = 'modular_out.mp4'
    save_csv: bool = False
    csv_filename: str = 'modular_data.csv'
    output_freq: int = 20 

    @property
    def nulb(self):
        return self.uLB * self.L_char / self.Re
    
    @property
    def omega(self):
        return 1.0 / (3.0 * self.nulb + 0.5)
