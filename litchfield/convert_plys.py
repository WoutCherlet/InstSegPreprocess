import os
import glob

from plyfile import PlyData, PlyElement



def regen_plys(parent_folder):


    for file in glob.glob(os.path.join(parent_folder, "**", "Augustus", "*.ply"), recursive=True):
        print(file)

        plydata = PlyData.read(file)
        vertices = plydata['vertex'].data
        new_element = PlyElement.describe(vertices, 'vertex')
        PlyData([new_element]).write(file)

    return

def main():

    PARENT_FOLDER = "/media/wcherlet/Stor1/wout/data/Litchfield/2019_ElizaSteffen_thesis/Bomen"
    
    regen_plys(PARENT_FOLDER)

    return


if __name__ == "__main__":
    main()