![Black Liquid Minimalist Daily Quotes LinkedIn Banner](https://github.com/user-attachments/assets/2cea33ba-a7e1-48c5-9e9d-8726eeb23a31)

Machine Learning subject from MS, Artificial Intelligence UNED

![GitHub last commit](https://img.shields.io/github/last-commit/TheDandyCodes/Machine-Learning)
![GitHub repo size](https://img.shields.io/github/repo-size/TheDandyCodes/Machine-Learning)
![GitHub issues](https://img.shields.io/github/issues/TheDandyCodes/Machine-Learning)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
This repository contains the Machine Learning subject from the MS in Artificial Intelligence at UNED. It includes various projects and exercises developed in Python and Jupyter Notebook.

## Features
- Comprehensive machine learning tutorials
- Practical examples and projects
- Jupyter Notebooks for interactive learning

## Installation
To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/TheDandyCodes/Machine-Learning.git
cd Machine-Learning
pip install -r requirements.txt
```

## Usage

---

# Ejercicios PrÃ¡cticos e Interactivos para la PresentaciÃ³n de LLMs

## ðŸŽ¯ **Ejercicios por SecciÃ³n**

### **SECCIÃ“N 1: IntroducciÃ³n - Ejercicio de Expectativas**

#### **Actividad**: "Â¿QuÃ© Puede Hacer un LLM?"
- **DuraciÃ³n**: 3 minutos
- **Formato**: Lluvia de ideas grupal
- **Objetivo**: Conocer expectativas y experiencias previas

**Instrucciones**:
1. Pregunta a la audiencia: "Mencionen tareas que creen que puede hacer un modelo como ChatGPT"
2. Lista las respuestas en pantalla
3. Al final, marca âœ… las correctas y âŒ las incorrectas
4. Explica brevemente por quÃ© algunas no son posibles

**Respuestas esperadas** (Correctas âœ…):
- Escribir textos, emails, cartas
- Responder preguntas generales
- Traducir idiomas
- Resumir documentos
- Escribir cÃ³digo simple
- Crear listas y esquemas

**Conceptos errÃ³neos comunes** (Incorrectas âŒ):
- Navegar por internet en tiempo real
- Recordar conversaciones pasadas
- Acceder a informaciÃ³n actualizada al minuto
- Realizar cÃ¡lculos matemÃ¡ticos complejos sin errores

---

### **SECCIÃ“N 2: Arquitectura - Ejercicio de Componentes**

#### **Actividad**: "Construye tu Transformer"
- **DuraciÃ³n**: 5 minutos
- **Formato**: AnalogÃ­a colaborativa
- **Objetivo**: Entender la secuencia de procesamiento

**Materiales**: Tarjetas con analogÃ­as (fÃ­sicas o virtuales)

**Instrucciones**:
1. Divide a la audiencia en grupos de 4-5 personas
2. Da a cada grupo las siguientes "profesiones":
   - Traductor (Embedding)
   - Estudiante (Encoder)
   - Director de orquesta (Attention)
   - Calculadora humana (Feed Forward)
   - Escritor (Decoder)
   - Juez (Output Layer)

3. Pide que organicen el orden correcto para procesar la frase: "Â¿CuÃ¡l es la capital de Francia?"

**Secuencia correcta**:
1. **Traductor** convierte palabras a nÃºmeros
2. **Estudiante** lee y comprende el contexto
3. **Director** decide quÃ© palabras son importantes
4. **Calculadora** procesa la informaciÃ³n
5. **Escritor** formula la respuesta
6. **Juez** selecciona la mejor palabra (ParÃ­s)

---

### **SECCIÃ“N 3: Mecanismo de AtenciÃ³n - Ejercicio Visual**

#### **Actividad**: "SÃ© el Mecanismo de AtenciÃ³n"
- **DuraciÃ³n**: 7 minutos
- **Formato**: SimulaciÃ³n grupal
- **Objetivo**: Entender cÃ³mo funciona la atenciÃ³n

**Ejemplo prÃ¡ctico**:
**Frase**: "El perro de MarÃ­a que vive en Madrid es muy inteligente"
**Pregunta**: "Â¿CÃ³mo es el perro?"

**Instrucciones**:
1. Muestra la frase en pantalla
2. Pide a la audiencia que identifique las palabras MÃS importantes para responder
3. Pide que califiquen cada palabra del 1-10 en importancia
4. Compara con el resultado real de un modelo de atenciÃ³n

**Resultados esperados**:
- "inteligente" â†’ 10 (respuesta directa)
- "perro" â†’ 9 (sujeto principal)
- "es" â†’ 6 (conecta sujeto con caracterÃ­stica)
- "muy" â†’ 5 (intensificador)
- "El", "de", "que", etc. â†’ 2-3 (palabras funcionales)

**DemostraciÃ³n**: Mostrar mapa de calor real de atenciÃ³n del modelo

---

### **SECCIÃ“N 4: Tipos de Modelos - Ejercicio de ClasificaciÃ³n**

#### **Actividad**: "Â¿QuÃ© Modelo UsarÃ­as?"
- **DuraciÃ³n**: 6 minutos
- **Formato**: Casos prÃ¡cticos
- **Objetivo**: Distinguir cuÃ¡ndo usar cada tipo de modelo

**Casos a resolver**:

1. **Caso 1**: "Necesitas clasificar emails como spam o no spam"
   - **Respuesta**: BERT (Encoder-only) - ClasificaciÃ³n
   
2. **Caso 2**: "Quieres generar un cuento infantil automÃ¡ticamente"
   - **Respuesta**: GPT (Decoder-only) - GeneraciÃ³n

3. **Caso 3**: "Necesitas traducir documentos del inglÃ©s al espaÃ±ol"
   - **Respuesta**: T5 (Encoder-Decoder) - TransformaciÃ³n

4. **Caso 4**: "Quieres analizar el sentimiento de reseÃ±as de productos"
   - **Respuesta**: BERT (Encoder-only) - AnÃ¡lisis

5. **Caso 5**: "Necesitas un chatbot conversacional"
   - **Respuesta**: GPT (Decoder-only) - DiÃ¡logo

**MetodologÃ­a**:
- Presenta cada caso
- Da 30 segundos para votar (levantando manos A, B, o C)
- Explica la respuesta correcta y por quÃ©

---

### **SECCIÃ“N 5: Aplicaciones - Ejercicio de Creatividad**

#### **Actividad**: "Inventa una AplicaciÃ³n"
- **DuraciÃ³n**: 8 minutos
- **Formato**: Brainstorming en equipos
- **Objetivo**: Pensar en aplicaciones innovadoras

**Instrucciones**:
1. Divide en grupos de 3-4 personas
2. Cada grupo recibe un sector:
   - Salud
   - EducaciÃ³n
   - Entretenimiento
   - Empresas
   - Gobierno
   - Arte y Creatividad

3. Pide que inventen una aplicaciÃ³n especÃ­fica de LLMs para su sector
4. Cada grupo presenta en 1 minuto

**Criterios de evaluaciÃ³n**:
- Â¿Es tÃ©cnicamente factible?
- Â¿Resuelve un problema real?
- Â¿Es innovador?

**Ejemplos de respuestas esperadas**:
- **Salud**: Asistente para redactar historiales mÃ©dicos
- **EducaciÃ³n**: Tutor personalizado que adapta explicaciones
- **Entretenimiento**: Generador de diÃ¡logos para videojuegos
- **Empresas**: Analizador automÃ¡tico de contratos
- **Gobierno**: Simplificador de lenguaje legal para ciudadanos
- **Arte**: Colaborador creativo para escritores

---

## ðŸ§© **Ejercicios de VerificaciÃ³n de ComprensiÃ³n**

### **Mini Quiz Interactivo** (2 minutos entre secciones)

#### **Pregunta 1**: Â¿QuÃ© componente decide quÃ© palabras son importantes?
- A) Encoder
- B) Attention Mechanism âœ…
- C) Decoder
- D) Embedding

#### **Pregunta 2**: Â¿QuÃ© tipo de modelo es mejor para continuar historias?
- A) BERT (Encoder-only)
- B) GPT (Decoder-only) âœ…
- C) T5 (Encoder-Decoder)

#### **Pregunta 3**: Los embeddings convierten:
- A) NÃºmeros en palabras
- B) Palabras en nÃºmeros âœ…
- C) Ideas en emociones
- D) Preguntas en respuestas

---

## ðŸŽ­ **Demostraciones en Vivo**

### **DemostraciÃ³n 1**: ComparaciÃ³n de Respuestas
**Tiempo**: 3 minutos

**Pregunta**: "Explica la fotosÃ­ntesis en tÃ©rminos simples"

**Mostrar respuestas de**:
- GPT-3 (respuesta general)
- Claude (respuesta mÃ¡s estructurada)
- BERT con fine-tuning educativo (respuesta muy especÃ­fica)

**Objetivo**: Mostrar diferencias en estilo y enfoque

### **DemostraciÃ³n 2**: AtenciÃ³n Visual
**Tiempo**: 2 minutos

**Herramienta**: Visualizador de atenciÃ³n online
**Frase ejemplo**: "La economÃ­a mundial, que habÃ­a mostrado signos de recuperaciÃ³n, ahora enfrenta nuevos desafÃ­os"
**Tarea**: Mostrar cÃ³mo cambia la atenciÃ³n al hacer diferentes preguntas

---

## ðŸ”¬ **Experimentos Conceptuales**

### **Experimento 1**: "El TelÃ©fono Descompuesto de la IA"
**Objetivo**: Mostrar cÃ³mo los errores se propagan

**Proceso**:
1. Frase inicial: "Los gatos son animales domÃ©sticos populares"
2. Pedir al modelo que la parafrasee
3. Usar la parÃ¡frasis como entrada para otra parÃ¡frasis
4. Repetir 5 veces
5. Mostrar cÃ³mo se degrada la informaciÃ³n

### **Experimento 2**: "Sesgo en AcciÃ³n"
**Objetivo**: Mostrar limitaciones y sesgos

**Proceso**:
1. Probar la misma pregunta con diferentes formulaciones
2. "Describe a un CEO exitoso" vs "Describe a una CEO exitosa"
3. Analizar diferencias en las respuestas
4. Discutir implicaciones Ã©ticas

---

## ðŸŽ¯ **Actividades de SÃ­ntesis Final**

### **Ejercicio de Cierre**: "Elevator Pitch de LLMs"
- **DuraciÃ³n**: 5 minutos
- **Formato**: Individual luego compartir

**Instrucciones**:
1. "Imagina que tienes 30 segundos para explicar quÃ© es un LLM a tu abuela"
2. Cada persona escribe su explicaciÃ³n
3. Algunos voluntarios comparten
4. Votar por la explicaciÃ³n mÃ¡s clara

**Criterios de evaluaciÃ³n**:
- Claridad para no-expertos
- PrecisiÃ³n tÃ©cnica bÃ¡sica
- Uso de analogÃ­as efectivas

### **ReflexiÃ³n Final**: "Una Pregunta Pendiente"
**Formato**: AnÃ³nimo con post-its o chat

**Pregunta**: "Â¿QuÃ© es lo que mÃ¡s te gustarÃ­a saber sobre LLMs que no hayamos cubierto?"

**PropÃ³sito**: 
- Identificar brechas en la comprensiÃ³n
- Sugerir recursos adicionales
- Planificar sesiones futuras

---

## ðŸ“‹ **Checklist para el Presentador**

### **Antes de cada ejercicio**:
- [ ] Explicar objetivo claramente
- [ ] Dar tiempo especÃ­fico
- [ ] Asegurar que todos participen
- [ ] Tener respuestas modelo preparadas

### **Durante cada ejercicio**:
- [ ] Circular por grupos si es grupal
- [ ] Dar avisos de tiempo
- [ ] Estar disponible para preguntas
- [ ] Documentar respuestas interesantes

### **DespuÃ©s de cada ejercicio**:
- [ ] Resumir aprendizajes clave
- [ ] Conectar con conceptos anteriores
- [ ] Aclarar dudas surgidas
- [ ] Hacer transiciÃ³n a siguiente tema

---

*Estos ejercicios estÃ¡n diseÃ±ados para mantener la participaciÃ³n activa y reforzar conceptos clave mediante la prÃ¡ctica directa.*
