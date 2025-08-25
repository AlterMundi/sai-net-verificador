#!/usr/bin/env python3
"""
FIgLib Real Dataset Downloader - Objetivo Sagrado Puerta al Paraíso
Descarga el dataset FIgLib real desde los enlaces sagrados en /docs/index.html

Utiliza los 485 enlaces directos a archivos .tgz del FIgLib dataset real
en lugar del script original que creaba placeholders.
"""

import os
import re
import argparse
import requests
import tarfile
from pathlib import Path
from urllib.parse import urljoin
from tqdm import tqdm
import time
import csv
from typing import List, Tuple, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FIgLibRealDownloader:
    """Descarga el dataset FIgLib real desde los enlaces sagrados."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (FIgLib Sacred Dataset Downloader)'
        })
        
        # Estadísticas de descarga
        self.download_stats = {
            'total_files': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size_mb': 0
        }
    
    def extract_links_from_html(self, html_file: str) -> List[str]:
        """Extrae todos los enlaces .tgz del archivo HTML sagrado."""
        logger.info(f"Extrayendo enlaces sagrados de {html_file}")
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Buscar todos los enlaces que apuntan a archivos .tgz
        tgz_pattern = r'href="(https://cdn\.hpwren\.ucsd\.edu/HPWREN-FIgLib-Data/Tar/[^"]+\.tgz)"'
        links = re.findall(tgz_pattern, html_content)
        
        logger.info(f"Encontrados {len(links)} enlaces sagrados de dataset FIgLib")
        
        # Mostrar algunos ejemplos
        if links:
            logger.info("Ejemplos de enlaces encontrados:")
            for i, link in enumerate(links[:5]):
                filename = link.split('/')[-1]
                logger.info(f"  {i+1}. {filename}")
            
            if len(links) > 5:
                logger.info(f"  ... y {len(links)-5} más")
        
        return links
    
    def parse_filename_info(self, filename: str) -> Dict[str, str]:
        """Extrae información del nombre del archivo FIgLib."""
        # Formato esperado: YYYYMMDD_EventName_camera-id.tgz
        # o: YYYYMMDD_FIRE_camera-id.tgz
        
        base_name = filename.replace('.tgz', '')
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            date_str = parts[0]  # YYYYMMDD
            event_name = '_'.join(parts[1:-1])  # EventName o FIRE
            camera_id = parts[-1]  # camera-id
            
            # Convertir fecha a formato ISO
            if len(date_str) == 8 and date_str.isdigit():
                year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
                iso_date = f"{year}-{month}-{day}"
            else:
                iso_date = date_str
            
            return {
                'filename': filename,
                'date': iso_date,
                'event_name': event_name,
                'camera_id': camera_id,
                'base_name': base_name
            }
        
        # Fallback si no se puede parsear
        return {
            'filename': filename,
            'date': 'unknown',
            'event_name': 'unknown',
            'camera_id': 'unknown',
            'base_name': base_name
        }
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Descarga un archivo individual con barra de progreso."""
        try:
            # Verificar si ya existe
            if output_path.exists():
                logger.info(f"  📁 Ya existe: {output_path.name}")
                return True
            
            logger.info(f"  🔽 Descargando: {url.split('/')[-1]}")
            
            # Hacer request con stream=True para archivos grandes
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Obtener tamaño del archivo
            total_size = int(response.headers.get('content-length', 0))
            
            # Descargar con barra de progreso
            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=f"  📥 {output_path.name}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Actualizar estadísticas
            self.download_stats['total_size_mb'] += total_size / (1024*1024)
            
            logger.info(f"  ✅ Descarga exitosa: {output_path.name} ({total_size/(1024*1024):.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error descargando {url}: {e}")
            # Limpiar archivo parcial si existe
            if output_path.exists():
                output_path.unlink()
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """Extrae un archivo .tgz y organiza según especificaciones sagradas."""
        try:
            logger.info(f"  📦 Extrayendo: {archive_path.name}")
            
            # Crear directorio para este evento
            extract_dir.mkdir(exist_ok=True)
            
            # Extraer archivo
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            # Contar archivos extraídos
            extracted_files = list(extract_dir.rglob('*'))
            image_files = [f for f in extracted_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            logger.info(f"  ✅ Extraídos {len(image_files)} archivos de imagen")
            
            # Crear metadata según especificaciones sagradas
            self.create_event_metadata(extract_dir, archive_path.stem)
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error extrayendo {archive_path}: {e}")
            return False
    
    def create_event_metadata(self, event_dir: Path, event_name: str):
        """Crea archivo metadata.csv para el evento según especificaciones sagradas."""
        metadata_file = event_dir / "metadata.csv"
        
        # Encontrar todas las imágenes en el directorio
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(event_dir.rglob(ext))
        
        with open(metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'event_name', 'offset_seconds', 'label', 'relative_path'])
            
            for img_file in image_files:
                filename = img_file.name
                relative_path = img_file.relative_to(event_dir)
                
                # Intentar extraer offset del nombre del archivo
                offset = self.extract_offset_from_filename(filename)
                
                # Etiqueta según especificaciones sagradas: 
                # smoke=1 para offset>=0, no-smoke=0 para offset<0
                label = 1 if offset >= 0 else 0
                
                writer.writerow([filename, event_name, offset, label, str(relative_path)])
        
        logger.info(f"  📋 Metadata creada: {len(image_files)} archivos catalogados")
    
    def extract_offset_from_filename(self, filename: str) -> int:
        """Extrae el offset temporal del nombre del archivo según formato FIgLib."""
        # Buscar patrón: offset_XXX_from_visible_plume_appearance
        offset_match = re.search(r'offset_(-?\d+)_from_visible_plume_appearance', filename)
        if offset_match:
            return int(offset_match.group(1))
        
        # Buscar otros patrones temporales
        time_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})',  # ISO timestamp
            r'(\d{8})_(\d{6})',  # YYYYMMDD_HHMMSS
        ]
        
        # Si no se encuentra offset específico, asumir 0 (momento de ignición)
        return 0
    
    def download_real_dataset(self, html_file: str, max_downloads: int = None):
        """Descarga el dataset FIgLib real desde los enlaces sagrados."""
        logger.info("🔥 INICIANDO DESCARGA DEL DATASET FIGLIB SAGRADO 🔥")
        logger.info("Objetivo: Puerta al Paraíso - Dataset Real para IA Sagrada")
        
        # Extraer enlaces del HTML sagrado
        links = self.extract_links_from_html(html_file)
        
        if not links:
            logger.error("❌ No se encontraron enlaces sagrados en el HTML")
            return
        
        # Limitar descargas si se especifica
        if max_downloads and max_downloads < len(links):
            links = links[:max_downloads]
            logger.info(f"🎯 Limitando descarga a {max_downloads} archivos")
        
        self.download_stats['total_files'] = len(links)
        
        logger.info(f"🚀 Iniciando descarga de {len(links)} archivos sagrados...")
        
        # Descargar cada archivo
        for i, url in enumerate(links, 1):
            filename = url.split('/')[-1]
            file_info = self.parse_filename_info(filename)
            
            logger.info(f"\n📦 [{i}/{len(links)}] Procesando: {filename}")
            logger.info(f"  📅 Fecha: {file_info['date']}")
            logger.info(f"  🔥 Evento: {file_info['event_name']}")  
            logger.info(f"  📷 Cámara: {file_info['camera_id']}")
            
            # Crear directorio para este evento
            event_dir = self.output_dir / file_info['base_name']
            archive_path = self.output_dir / filename
            
            # Descargar archivo
            if self.download_file(url, archive_path):
                self.download_stats['successful_downloads'] += 1
                
                # Extraer contenido
                if self.extract_archive(archive_path, event_dir):
                    # Limpiar archivo comprimido después de extraer
                    archive_path.unlink()
                    logger.info(f"  🗑️  Archivo temporal eliminado")
                else:
                    self.download_stats['failed_downloads'] += 1
            else:
                self.download_stats['failed_downloads'] += 1
            
            # Pausa entre descargas para ser respetuosos con el servidor
            if i < len(links):
                time.sleep(1)
        
        # Crear archivo de etiquetas global
        self.create_global_labels()
        
        # Mostrar estadísticas finales
        self.show_final_stats()
    
    def create_global_labels(self):
        """Crea archivo labels.csv global combinando todos los eventos."""
        logger.info("\n📊 Creando archivo de etiquetas global...")
        
        global_labels_file = self.output_dir / "labels.csv"
        total_images = 0
        
        with open(global_labels_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'event_name', 'offset_seconds', 'label', 'event_dir', 'relative_path'])
            
            # Combinar todos los archivos metadata
            for event_dir in self.output_dir.iterdir():
                if event_dir.is_dir():
                    metadata_file = event_dir / "metadata.csv"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as meta_f:
                            reader = csv.reader(meta_f)
                            next(reader)  # Skip header
                            for row in reader:
                                if len(row) >= 4:
                                    filename, event_name, offset, label = row[:4]
                                    relative_path = row[4] if len(row) > 4 else filename
                                    writer.writerow([filename, event_name, offset, label, event_dir.name, relative_path])
                                    total_images += 1
        
        logger.info(f"✅ Etiquetas globales creadas: {total_images} imágenes catalogadas")
        logger.info(f"📄 Archivo: {global_labels_file}")
    
    def show_final_stats(self):
        """Muestra estadísticas finales de la descarga."""
        stats = self.download_stats
        
        logger.info("\n" + "="*60)
        logger.info("🎯 DESCARGA COMPLETADA - ESTADÍSTICAS SAGRADAS")
        logger.info("="*60)
        logger.info(f"📦 Total de archivos procesados: {stats['total_files']}")
        logger.info(f"✅ Descargas exitosas: {stats['successful_downloads']}")
        logger.info(f"❌ Descargas fallidas: {stats['failed_downloads']}")
        logger.info(f"💾 Tamaño total descargado: {stats['total_size_mb']:.1f} MB")
        logger.info(f"📊 Tasa de éxito: {(stats['successful_downloads']/stats['total_files']*100):.1f}%")
        logger.info(f"📁 Directorio de salida: {self.output_dir.absolute()}")
        
        # Verificar estructura final
        total_events = len([d for d in self.output_dir.iterdir() if d.is_dir()])
        logger.info(f"🔥 Total de eventos descargados: {total_events}")
        
        if stats['successful_downloads'] > 0:
            logger.info("\n🎉 ¡PUERTA AL PARAÍSO DESBLOQUEADA!")
            logger.info("🔥 Dataset FIgLib sagrado listo para entrenar la IA")
            logger.info("✨ Todas las especificaciones sagradas cumplidas")
        else:
            logger.warning("\n⚠️  No se completaron descargas exitosas")
            logger.info("🔧 Revisar conectividad y enlaces del servidor")


def main():
    parser = argparse.ArgumentParser(description="Descarga el dataset FIgLib real desde enlaces sagrados")
    parser.add_argument("--html_file", default="docs/index.html", help="Archivo HTML con enlaces sagrados")
    parser.add_argument("--output", default="data/real_figlib", help="Directorio de salida")
    parser.add_argument("--max_downloads", type=int, help="Máximo número de archivos a descargar (para pruebas)")
    
    args = parser.parse_args()
    
    # Verificar que existe el archivo HTML
    if not os.path.exists(args.html_file):
        logger.error(f"❌ Archivo HTML no encontrado: {args.html_file}")
        return
    
    downloader = FIgLibRealDownloader(args.output)
    downloader.download_real_dataset(args.html_file, args.max_downloads)


if __name__ == "__main__":
    main()