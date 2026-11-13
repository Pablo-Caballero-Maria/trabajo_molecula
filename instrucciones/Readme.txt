// Desde una terminal de la máquina local (instalación nativa) instalar:
apt-get update
apt-get install wget
mkdir /home/mongodata
cd /home/mongodata

// Se descarga la copia de Seguridad de la BBDD:
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GmjgN18_ZN8Xn25WPCb2VHGWtQB90QI1' -O dump.tar

// Se restaura la BBDD utilizando la copia de seguridad 
// Para local
mongorestore mongodb://adp:adp@localhost:27017/?authSource=admin&readPreference=primary&ssl=false&directConnection=true dump

//para mongo Atlas buscar cadena de conexión en "Cmd Line Tools"
mongorestore --uri mongodb+srv://gcerruela:<PASSWORD>@cluster0.5m0pk.mongodb.net 




