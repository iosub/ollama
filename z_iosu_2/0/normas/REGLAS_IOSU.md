# REGLAS IMPORTANTES - NO OLVIDAR

## ❌ PROHIBIDO

1. **NO compilar automáticamente** - El usuario compila manualmente
2. **NO ejecutar comandos de build** sin permiso explícito del usuario
3. **NO ejecutar `powershell -ExecutionPolicy Bypass -File ... build_windows.ps1`** 
4. **NO asumir que debo compilar** cuando el usuario dice "ya lo compilo" o similar

## ✅ PERMITIDO

1. **Preparar comandos** para que el usuario los ejecute
2. **Analizar logs** de compilación o ejecución
3. **Crear/modificar archivos** de código, parches, documentación
4. **Leer archivos** para investigar problemas
5. **Sugerir** acciones, pero NO ejecutarlas sin confirmación

## 📋 RECORDATORIOS

- Cuando el usuario dice "ya lo compilo" → significa que ÉL está compilando
- La compilación tarda varios minutos → NO interrumpir
- Logs de test en: `z_iosu_2\logs3\q2_XX.log`
- Parches en: `llama\patches\00XX-*.patch`

## 🎯 WORKFLOW TÍPICO

1. Usuario identifica problema
2. Asistente investiga y propone solución
3. Asistente crea/modifica archivos necesarios
4. **USUARIO compila** (NO el asistente)
5. **USUARIO ejecuta tests** (NO el asistente)
6. Asistente analiza resultados

---
**Última actualización**: 2025-11-28
**Contexto**: Proyecto Qwen3-VL split models en Ollama
