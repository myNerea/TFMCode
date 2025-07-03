const link = document.createElement('link');
link.rel = 'icon';
link.id = 'favicon';
link.type = 'image/png';  // si usas .ico cambia a 'image/x-icon'
link.href = '/logo_us.png';  // asegúrate que está en la carpeta public/
document.head.appendChild(link);
