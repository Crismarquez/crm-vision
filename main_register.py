
import time
import pickle

import numpy as np

from visionanalytic.recognition import FaceRecognition, SequentialRecognition
from visionanalytic.framer import Register
from config.config import RESULTS_DIR

USERS_RESULTS_DIR = RESULTS_DIR / "new_users"
USERS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRM_DIR = RESULTS_DIR / "crm_vision.npy"

if not CRM_DIR.exists():
    np.save(CRM_DIR, [], allow_pickle=True)

if __name__ == "__main__":
    print("\n")
    print("*"*30)
    print("Bienvenido al registro de crm-vision  \n")
    print("*"*30)
    print("\n")
    print("Configuraciones:  \n")

    source = input("Habilitar el puerto número: ")
    source = int(source)

    sequential = SequentialRecognition("mean")

    register_framer = Register(
        source=source,
        recognition=FaceRecognition(det_size=(320, 320)),
        sequential_model=sequential,
        frame_skipping=5,
        write=True
    )

    new_user = input("Enter new user (y/n): ")

    while new_user == "y":

        name = input("Por favor ingrese su nombre: ")
        age = input("Por favor ingrese su edad: ")
        phone = input("Por favor ingrese su numero de teléfono: ")
        id = input("Por favor ingrese su numero de identificación: ")
        accept = input("Acepta terminos y condiciones (y/n): ")

        time_register = time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())

        user_info = {
            "name": name,
            "age": age,
            "phone": phone,
            "id": id,
            "accept": accept
        }
        print(user_info)

        file_name = f"{user_info['id']}_{time_register}_user_register.pkl"
        user_info = register_framer.capture(user_info)

        # save all register 
        with open(USERS_RESULTS_DIR / file_name, "wb") as handle:
            pickle.dump(user_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # delete meta data
        user_info.pop("meta_data")

        # update crm
        crm_npy = np.load(CRM_DIR, allow_pickle=True)
        crm_npy = list(crm_npy) + [user_info]
        np.save(CRM_DIR, np.array(crm_npy), allow_pickle=True)

        new_user = input("Enter new user (y/n):")

