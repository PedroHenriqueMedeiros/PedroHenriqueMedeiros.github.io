/* Definição de todas as máscaras dos filtros */
float media[] = {
    1,  1,  1,
    1,  1,  1,
    1,  1,  1
};
float gauss[] = {
    1,  2,  1,
    2,  4,  2,
    1,  2,  1
};
float horizontal[]={
    -1,  0,  1,
    -2,  0,  2,
    -1,  0,  1
};
float vertical[]={
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};
float laplacian[]={
     0, -1,  0,
    -1,  4, -1,
     0, -1,  0
};

/* Resultado da convolução digital dos filtros gauss e laplacian acima.
 * Ela pode ser feita à mão ou utilizando alguma ferramenta como Simulink
 *   ou Python + numpy+ scipy (exemplo em anexo). 
 */
float laplgauss[]={
    0, -1, -2, -1,  0,
   -1,  0,  2,  0, -1,
   -2,  2,  8,  2, -2,
   -1,  0,  2,  0, -1,
    0, -1, -2, -1,  0,
};
