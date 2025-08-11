#ifndef QUANT8_H
#define QUANT8_H

#include <stdint.h>

/* Esta implementación utiliza cuantizacion de 8 bits con signo. Utiliza punto fijo de 3 bits para
bias de cuantizacion. La precision es ajustable con las constantes definidas Q_INT_BITS y Q_FRAC_BITS */

// Configuración dinámica (ajusta estos valores según tu hardware)
#define Q_INT_BITS  15      // Bits para parte entera
#define Q_FRAC_BITS 17      // Bits para parte fraccional

// Verificación de rangos
#if (Q_INT_BITS + Q_FRAC_BITS) != 32
    #error "Qn.m debe sumar 32 bits (int32_t)"
#endif

typedef int8_t quant8;  // Tipo cuantizado (8 bits)

// Parámetros de cuantización (ahora con fixed-point configurable)
typedef struct {
    int32_t scale;       // Escala en formato Qn.m (configurable)
    //int32_t inv_scale;    // 1/scale (precalculado)
    int8_t zero_point;    // Punto cero
} qparam_t;

// Macros básicas
#define Q_SCALE      (1 << Q_FRAC_BITS)
#define Q_MAX_VAL    ((1 << (Q_INT_BITS - 1)) - 1)  // Rango máximo
#define Q_MIN_VAL    (-(1 << (Q_INT_BITS - 1)))     // Rango mínimo

// Nuevos rangos para int8
#define Q_MIN (-128)
#define Q_MAX 127

//////////////////////////////////// Macros de conversion ////////////////////////////////////
#define Q_CLAMP(qv) ( (qv > Q_MAX) ? Q_MAX : ( (qv < Q_MIN) ? Q_MIN : (quant8)qv ) )

#define QUANTIZE(value, qp) Q_CLAMP( (int)(roundf(qp.zero_point + (value / qp.scale)) )
#define DEQUANTIZE(qvalue, qp) ( (float)(qp.scale * ((float)qvalue - qp.zero_point)) )

//////////////////////////////////// Funciones ////////////////////////////////////////
// Conversiones
static inline int32_t float_to_q(float f) {
    return (int32_t)(f * Q_SCALE);
}

static inline float q_to_float(int32_t q) {
    return (float)q / Q_SCALE;
}

// Multiplicación con ajuste de formato
static inline int32_t q_mul(int32_t a, int32_t b) {
    return ((int64_t)a * b) >> Q_FRAC_BITS;
}

// Suma con saturación
static inline int32_t q_add(int32_t a, int32_t b) {
    int64_t tmp = (int64_t)a + b;
    if (tmp > Q_MAX_VAL) return Q_MAX_VAL;
    if (tmp < Q_MIN_VAL) return Q_MIN_VAL;
    return (int32_t)tmp;
}


#ifdef __cplusplus
extern "C" {
#endif

    // Calcula parámetros de cuantización
    void quantize_param(float *values, int size, qparam_t *qp);

    // Cuantiza/descuantiza vectores
    void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp);
    void dequantize_vec(quant8 qvalues[], float values[], int size, qparam_t qp);

    // Operaciones combinadas (MAC)
    int32_t mul_add_vec(quant8 a[], qparam_t qa, quant8 b[], qparam_t qb, int size);

#ifdef __cplusplus
}
#endif

#endif
