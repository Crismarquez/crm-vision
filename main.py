import cv2


if __name__ == "__main__":
    print("\n")
    print("*"*30)
    print("Bienvenido a crm-vision  \n")
    print("*"*30)
    print("\n")
    print("Configuraciones:  \n")

    source = input("Habilitar el puerto número: ")
    source = int(source)

    cap = cv2.VideoCapture(source)

    while True:

        rep, frame = cap.read()

        if not rep:
            break

        cv2.imshow("crm-vision", frame)

        if cv2.waitKey(10) == ord("q"):
                    break

    cap.realease()
    cv2.destroyAllWindows()