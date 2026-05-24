# Presentación: Sistema difuso para **Blackjack-v1**

**Alumno:** _(tu nombre)_  
**Materia:** Lógica difusa  
**Entorno:** Gymnasium — Toy Text — `Blackjack-v1`  
**Código:** `blackjack_fis.py`

---

## 1. Entorno elegido

Elegí **Blackjack** de la sección **Toy Text** porque:

- Es distinto a ejemplos típicos de Control o Box2D.
- Las observaciones se interpretan de forma natural en lenguaje humano (“suma alta”, “crupier fuerte”).
- Solo hay **dos acciones**, lo que facilita un FIS claro para la exposición.

```python
import gymnasium as gym
env = gym.make("Blackjack-v1")
```

---

## 2. Dinámica del entorno

### Observación `(suma_jugador, carta_crupier, as_utilizable)`

| Componente | Rango | Significado |
|------------|-------|-------------|
| `suma_jugador` | 4–21 | Puntos actuales del jugador |
| `carta_crupier` | 1–10 | Carta visible del crupier (1 = As) |
| `as_utilizable` | 0 o 1 | Si hay un As contando como 11 sin pasarse |

### Acciones (espacio discreto)

| Valor | Nombre | Efecto |
|-------|--------|--------|
| **0** | **Stick** (plantarse) | Termina el turno del jugador |
| **1** | **Hit** (pedir) | Recibe otra carta |

### Objetivo / qué simula

Simula una partida simplificada de **veintiuno** contra un crupier con baraja infinita. El jugador intenta **acercarse a 21 sin pasarse** y **superar al crupier** al final de la mano.

**Recompensas:** +1 ganar, −1 perder, 0 empate.

---

## 3. Diseño del sistema difuso

| Parámetro | Elección |
|-----------|----------|
| **Tipo** | Mamdani |
| **Operador AND** | `min` (por defecto en scikit-fuzzy) |
| **Implicación** | Recorte del consecuente |
| **Agregación** | Máximo |
| **Defuzzificación** | **Centroide** |
| **Decisión final** | Si centroide ≥ 0.5 → HIT (1), si no → STICK (0) |

---

## 4. Variables lingüísticas de ENTRADA

### 4.1 `suma_jugador` (universo: 4–21)

| Término | Función | Justificación |
|---------|---------|---------------|
| `hasta_11` | Triangular `[4, 11, 11]` | Mano muy baja; casi siempre conviene pedir. |
| `es_12` | Triangular `[12, 12, 12]` | Caso crítico de la estrategia básica. |
| `de_13_a_16` | Triangular `[13, 14.5, 16]` | Zona “difícil” (riesgo de pasarse vs crupier). |
| `desde_17` | Triangular `[17, 19, 21]` | Mano fuerte; suele plantarse. |
| `soft_hasta_17` | Triangular `[12, 14, 17]` | Mano blanda baja/media. |
| `soft_18` | Triangular `[18, 18, 18]` | Soft 18: decisión según crupier. |
| `soft_19_mas` | Triangular `[19, 20, 21]` | Mano blanda alta. |

**¿Por qué triangulares?** Permiten **solapamiento** entre regiones (p. ej. 12 y 13–16), de modo que varias reglas aportan gradualmente al resultado — comportamiento típico Mamdani.

---

### 4.2 `carta_crupier` (universo: 1–10)

| Término | Función | Justificación |
|---------|---------|---------------|
| `d_2_3` | Triangular `[2, 2.5, 3]` | Crupier débil en el sentido de que el jugador con 12 aún pide. |
| `d_4_5_6` | Triangular `[4, 5, 6]` | Crupier “malo”: probabilidad alta de que se pase. |
| `d_2_a_6` | Triangular `[2, 4, 6]` | Para plantarse con 13–16. |
| `d_7_a_10` | Triangular `[7, 8.5, 10]` | Crupier fuerte. |
| `d_9_10_A` | Triangular `[9, 10, 10]` | Muy fuerte (soft 18). |
| `d_2_7_8` | Triangular `[2, 7, 8]` | Plantarse en soft 18. |
| `as_visible` | Triangular `[1, 1, 1]` | Término **puntual** para As=1 en universo discreto. |

**¿Por qué triangulares y un puntual?** El universo es **entero**. Los triángulos cubren rangos; el término `as_visible` evita huecos en el valor 1 (As), que con trapezoides dejó estados sin reglas activas.

---

### 4.3 `as_utilizable` (universo: 0, 1)

| Término | Función | Justificación |
|---------|---------|---------------|
| `no` | Triangular `[0, 0, 0]` | Mano dura. |
| `si` | Triangular `[1, 1, 1]` | Mano blanda (As como 11). |

**¿Por qué casi singleton?** Variable **binaria**; no hay grados intermedios reales, pero encaja en el marco Mamdani y separa reglas duras (HD*) de blandas (BL*).

---

## 5. Variable lingüística de SALIDA

### `inclinacion_pedir` ∈ [0, 1]

| Término | Función | Significado |
|---------|---------|-------------|
| `plantarse` | Triangular `[0, 0, 0.35]` | Tendencia a acción **0** |
| `pedir` | Triangular `[0.65, 1, 1]` | Tendencia a acción **1** |

**¿Por qué triangulares?** Son estándar en Mamdani para consecuentes difusos y permiten **defuzzificar por centroide** obteniendo un escalar interpretable antes de mapear a {0, 1}.

---

## 6. Reglas del sistema (y su razón)

### Mano dura (`as_utilizable = no`)

| ID | SI … | ENTONCES | Razón |
|----|------|----------|-------|
| **HD1** | suma `hasta_11` | `pedir` | Poco riesgo de pasarse. |
| **HD2** | suma `desde_17` | `plantarse` | Mano fuerte. |
| **HD3** | suma `es_12` Y crupier `d_4_5_6` | `plantarse` | Clásico 12 vs 4–6. |
| **HD4** | suma `es_12` Y crupier `d_2_3` | `pedir` | Aún conviene mejorar. |
| **HD5** | suma `es_12` Y crupier `d_7_a_10` | `pedir` | Crupier fuerte. |
| **HD6** | suma `es_12` Y `as_visible` | `pedir` | As del crupier es peligroso. |
| **HD7** | suma `de_13_a_16` Y crupier `d_2_a_6` | `plantarse` | Dejar que el crupier arriesgue. |
| **HD8** | suma `de_13_a_16` Y crupier `d_7_a_10` | `pedir` | Hay que mejorar la mano. |
| **HD9** | suma `de_13_a_16` Y `as_visible` | `pedir` | Igual que crupier fuerte. |

### Mano blanda (`as_utilizable = si`)

| ID | SI … | ENTONCES | Razón |
|----|------|----------|-------|
| **BL1** | suma `soft_hasta_17` | `pedir` | El As absorbe el exceso. |
| **BL2** | suma `soft_19_mas` | `plantarse` | Mano ya muy buena. |
| **BL3** | `soft_18` Y `d_2_7_8` | `plantarse` | Soft 18 razonable vs cartas débiles. |
| **BL4** | `soft_18` Y `d_4_5_6` | `pedir` | Mejorar contra crupier medio. |
| **BL5** | `soft_18` Y `d_9_10_A` | `pedir` | Crupier muy fuerte. |
| **BL6** | `soft_18` Y `as_visible` | `pedir` | As del crupier: seguir mejorando. |

Las reglas codifican la **estrategia básica** observada al simular: pedir con sumas bajas, plantarse con altas, y en 12–16 depender de la carta del crupier.

---

## 7. Prueba de inferencia (1–2 reglas)

**Comando:**

```bash
python blackjack_fis.py --demo-inferencia
```

### Caso A — Regla **HD1**

- **Entradas:** suma = 7, crupier = 10, as_util = 0  
- **Pertenencias:** `hasta_11(7) > 0`, `as_util = no`  
- **Centroide:** ≈ 0.81 → **HIT**  
- **Interpretación:** aunque el crupier muestra 10, con 7 aún es muy conveniente pedir.

### Caso B — Regla **HD3**

- **Entradas:** suma = 12, crupier = 5, as_util = 0  
- **Pertenencias:** `es_12(12) = 1`, `d_4_5_6(5) = 1`  
- **Centroide:** ≈ 0.15 → **STICK**  
- **Interpretación:** 12 vs 5 es el caso clásico de plantarse.

---

## 8. Defuzzificador: **centroide**

**Razón de la elección:**

1. Es el método **estándar en Mamdani** cuando el consecuente es difuso.
2. Combina todas las reglas activas en un **único valor** en [0, 1].
3. Es **fácil de explicar** en clase: “promedio ponderado del área de salida”.
4. Con umbral 0.5 se traduce de forma natural a las dos acciones discretas del entorno.

---

## 9. Corrida en vivo

```bash
python blackjack_fis.py --vivo 5 --render
```

Muestra cada paso en consola y la ventana **pygame** del entorno. Al terminar imprime estadísticas del juego y del FIS (centroides, % HIT/STICK).

**Evaluación cuantitativa:**

```bash
python blackjack_fis.py --eval 5000
python blackjack_fis.py --eval 50 --render
```

Con `--render` en `--eval` se abre pygame en los primeros 10 episodios (ajusta con `--render-max`); al final salen las mismas estadísticas difusas.

**Resultados de referencia (FIS difuso):**

| Métrica | Valor típico |
|---------|----------------|
| Recompensa media | ≈ −0.05 a −0.07 |
| Tasa de victoria | ≈ 42–43 % |
| Empates | ≈ 9 % |

*(Pegar aquí la salida de tu corrida real.)*

---

## 10. Conclusión

1. **Resultados:** _(pegar salida de `--eval`)_. El FIS toma decisiones coherentes según suma, carta del crupier y as utilizable.
2. **Experiencia de diseño:** definir términos lingüísticos y reglas a partir de la simulación; calibrar funciones de membresía en un universo discreto (As = 1, sumas 12–16).
3. **Aprendizaje:** el valor del ejercicio está en el diseño del FIS (Mamdani + centroide), no en maximizar la tasa de victoria del casino.

---

## Comandos rápidos para la exposición

```bash
pip install -r requirements.txt
python blackjack_fis.py --demo-inferencia
python blackjack_fis.py --vivo 3 --render
python blackjack_fis.py --eval 500
```
