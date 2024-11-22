import cv2
import numpy as np
import logging
from ultralytics import YOLO
import easyocr
import re
import tkinter as tk
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
import face_recognition

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar logging para suprimir los mensajes de YOLO
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Inicializar EasyOCR
reader = easyocr.Reader(['es'])  # Español para placas colombianas

# Inicializar YOLO para detección facial
face_model = YOLO('modelos/yolov8n-face.pt')

# Crear carpeta para almacenar datos
if not os.path.exists('parqueadero_data'):
    os.makedirs('parqueadero_data')

def detectar_placa(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 11, 17, 17)
    bordes = cv2.Canny(gris, 30, 200)
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]
    
    for contorno in contornos:
        aprox = cv2.approxPolyDP(contorno, 10, True)
        if len(aprox) == 4:
            return aprox
    return None

def procesar_placa(imagen, ubicacion):
    mask = np.zeros(imagen.shape[:2], np.uint8)
    cv2.drawContours(mask, [ubicacion], 0, 255, -1)
    return cv2.bitwise_and(imagen, imagen, mask=mask)

def es_placa_valida(texto):
    patron = r'^[A-Z]{3}\d{3}$'
    return re.match(patron, texto) is not None

def extract_face_encoding(face_image):
    if face_image is None:
        return None
    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_face)
    if encodings:
        return encodings[0]
    return None

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), 
                              height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Frame para los botones
        button_frame = tk.Frame(window)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_entrada = tk.Button(button_frame, text="Registrar Entrada", width=20, 
                                   command=self.iniciar_registro_entrada)
        self.btn_entrada.pack(side=tk.LEFT, padx=5)

        self.btn_salida = tk.Button(button_frame, text="Registrar Salida", width=20, 
                                   command=self.iniciar_registro_salida)
        self.btn_salida.pack(side=tk.RIGHT, padx=5)

        # Estado del sistema
        self.registrando = False
        self.modo_salida = False
        self.rostro_capturado = None
        self.placa_capturada = None
        self.mensaje_estado = ""
        self.color_mensaje = (0, 255, 0)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def iniciar_registro_entrada(self):
        logger.info("Iniciando registro de entrada")
        self.registrando = True
        self.modo_salida = False
        self.rostro_capturado = None
        self.placa_capturada = None
        self.mensaje_estado = "Registrando entrada... Espere por favor"
        self.color_mensaje = (0, 255, 0)
        self.btn_entrada.config(text="Registrando entrada...", state="disabled")
        self.btn_salida.config(state="disabled")

    def iniciar_registro_salida(self):
        logger.info("Iniciando registro de salida")
        self.registrando = True
        self.modo_salida = True
        self.rostro_capturado = None
        self.placa_capturada = None
        self.mensaje_estado = "Registrando salida... Espere por favor"
        self.color_mensaje = (0, 255, 0)
        self.btn_salida.config(text="Registrando salida...", state="disabled")
        self.btn_entrada.config(state="disabled")

    def guardar_registro(self, coincide=None, entrada_path=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tipo = 'salida' if self.modo_salida else 'entrada'
        
        # Guardar imagen del rostro
        rostro_path = f'parqueadero_data/rostro_{self.placa_capturada}{tipo}{timestamp}.jpg'
        cv2.imwrite(rostro_path, self.rostro_capturado)
        
        # Crear registro
        registro = {
            'placa': self.placa_capturada,
            'rostro_path': rostro_path,
            'timestamp': timestamp,
            'tipo': tipo
        }
        
        # Añadir información adicional para registros de salida
        if self.modo_salida and coincide is not None:
            registro.update({
                'coincidencia_rostro': coincide,
                'rostro_entrada_path': entrada_path,
                'autorizado': coincide
            })
        
        # Guardar registro en JSON
        filename = f'parqueadero_data/registro_{self.placa_capturada}{tipo}{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(registro, f)
        
        logger.info(f"Registro de {tipo} guardado - Placa: {self.placa_capturada}")
        return rostro_path

    def verificar_entrada_y_rostro(self):
        logger.info(f"Verificando entrada y rostro para la placa: {self.placa_capturada}")
        
        # Buscar registro de entrada más reciente
        entrada_registrada = None
        for filename in sorted(os.listdir('parqueadero_data'), reverse=True):
            if filename.endswith('.json') and filename.startswith(f'registro_{self.placa_capturada}entrada'):
                with open(os.path.join('parqueadero_data', filename), 'r') as f:
                    registro = json.load(f)
                    if registro['placa'] == self.placa_capturada and registro['tipo'] == 'entrada':
                        entrada_registrada = registro
                        logger.info(f"Registro de entrada encontrado: {filename}")
                        break
        
        if entrada_registrada is None:
            logger.warning(f"No se encontró registro de entrada para la placa {self.placa_capturada}")
            return False, None, None

        rostro_entrada_path = entrada_registrada['rostro_path']
        
        # Verificar rostros
        rostro_entrada = cv2.imread(rostro_entrada_path)
        if rostro_entrada is None:
            logger.error(f"No se pudo cargar la imagen del rostro de entrada: {rostro_entrada_path}")
            return False, None, rostro_entrada_path

        # Verificar que tengamos el rostro de salida
        if self.rostro_capturado is None:
            logger.error("No se ha capturado el rostro de salida")
            return False, None, rostro_entrada_path

        # Obtener encodings de ambos rostros
        encoding_entrada = extract_face_encoding(rostro_entrada)
        encoding_salida = extract_face_encoding(self.rostro_capturado)
        
        if encoding_entrada is None or encoding_salida is None:
            logger.error("No se pudieron extraer las características faciales de una o ambas imágenes")
            return False, None, rostro_entrada_path
        
        # Realizar la comparación
        match = face_recognition.compare_faces([encoding_entrada], encoding_salida)[0]
        distance = face_recognition.face_distance([encoding_entrada], encoding_salida)[0]
        
        logger.info(f"Coincidencia de rostro: {match}, Distancia: {distance}")
        
        # Umbral de similitud ajustado
        coincidencia = match and distance < 0.6
        return coincidencia, distance, rostro_entrada_path
    def mostrar_comparacion(self, rostro_path):
        if rostro_path is None or self.rostro_capturado is None:
            logger.warning("No se pueden comparar los rostros: falta información")
            return False

        rostro_registrado = cv2.imread(rostro_path)
        if rostro_registrado is None:
            logger.error(f"No se pudo cargar la imagen del rostro registrado: {rostro_path}")
            return False

        # Redimensionar imágenes para comparación
        altura_deseada = 300
        rostro_registrado = cv2.resize(rostro_registrado, (int(altura_deseada * rostro_registrado.shape[1] / rostro_registrado.shape[0]), altura_deseada))
        rostro_actual = cv2.resize(self.rostro_capturado, (int(altura_deseada * self.rostro_capturado.shape[1] / self.rostro_capturado.shape[0]), altura_deseada))
        
        # Crear imagen de comparación
        comparacion = np.hstack((rostro_registrado, rostro_actual))
        
        # Añadir etiquetas
        cv2.putText(comparacion, "Rostro Entrada", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparacion, "Rostro Salida", (rostro_registrado.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar comparación
        cv2.imshow("Comparación de rostros", comparacion)
        cv2.waitKey(3000)  # Mostrar por 3 segundos
        cv2.destroyAllWindows()
        return True

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_display = frame.copy()

            if self.registrando:
                # Detección de placa primero
                if self.placa_capturada is None:
                    ubicacion = detectar_placa(frame)
                    if ubicacion is not None:
                        placa = procesar_placa(frame, ubicacion)
                        resultado = reader.readtext(placa)
                        if resultado:
                            texto = resultado[0][-2]
                            texto = ''.join(e for e in texto if e.isalnum()).upper()
                            if es_placa_valida(texto):
                                self.placa_capturada = texto
                                cv2.drawContours(frame_display, [ubicacion], -1, (0, 255, 0), 3)
                                cv2.putText(frame_display, texto, (ubicacion[0][0][0], ubicacion[0][0][1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                logger.info(f"Placa capturada: {self.placa_capturada}")

                # Detección de rostro después de la placa
                if self.placa_capturada is not None and self.rostro_capturado is None:
                    results = face_model(frame, verbose=False)
                    for result in results[0].boxes:
                        x1, y1, x2, y2 = map(int, result.xyxy[0])
                        self.rostro_capturado = frame[y1:y2, x1:x2].copy()
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_display, "Rostro capturado", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        logger.info("Rostro capturado")
                        break

                # Proceso de verificación y registro
                if self.rostro_capturado is not None and self.placa_capturada is not None:

                    if self.modo_salida:
                        # Realizar verificación
                        coincide, distancia, rostro_entrada_path = self.verificar_entrada_y_rostro()
                        
                        # Mostrar comparación visual
                        self.mostrar_comparacion(rostro_entrada_path)
                        
                        # Guardar registro de salida
                        rostro_salida_path = self.guardar_registro(coincide=coincide, 
                                                                 entrada_path=rostro_entrada_path)
                        
                        # Actualizar mensaje de estado
                        if coincide:
                            self.mensaje_estado = f"SALIDA AUTORIZADA - Placa: {self.placa_capturada}"
                            self.color_mensaje = (0, 255, 0)  # Verde
                        else:
                            self.mensaje_estado = f"SALIDA NO AUTORIZADA - Placa: {self.placa_capturada}"
                            self.color_mensaje = (0, 0, 255)  # Rojo
                        
                        logger.info(self.mensaje_estado)
                        
                    else:
                        # Proceso de entrada normal
                        self.guardar_registro()
                        self.mensaje_estado = f"Entrada registrada - Placa: {self.placa_capturada}"
                        self.color_mensaje = (0, 255, 0)
                    
                    # Resetear el proceso
                    self.registrando = False
                    self.btn_entrada.config(text="Registrar Entrada", state="normal")
                    self.btn_salida.config(text="Registrar Salida", state="normal")

            # Mostrar mensaje de estado
            if self.mensaje_estado:
                cv2.putText(frame_display, self.mensaje_estado, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_mensaje, 2)

            # Actualizar la imagen en la interfaz
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def _del_(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    try:
        # Configurar el estilo de la ventana
        root = tk.Tk()
        root.title("Sistema de Control de Parqueadero")
        
        # Configurar el tamaño y posición de la ventana
        window_width = 800
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Iniciar la aplicación
        app = App(root, "Sistema de Control de Parqueadero")
        
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {str(e)}")
        raise
