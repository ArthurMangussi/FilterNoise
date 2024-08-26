# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from configparser import ConfigParser
# =================================================================================================
class EmailMestrado:
    """Classe para envio de um e-mail para avisar quando terminou de rodar o código completo
    da imputação dos algoritmos em 20 datasets"""

    # ---------------------------------------------------------------------------------------------
    def __init__(self):
        config = ConfigParser()
        config.read("C:\\Users\\Mult-e\\Desktop\\@Codigos\\email.ini")

        # Configurações do servidor SMTP
        self.SMTP_SERVER = config["SMTP"]["SMTP_SERVER"]
        self.SMTP_PORT = config["SMTP"]["SMTP_PORT"]
        self.SMTP_USER = config["SMTP"]["SMTP_USER"]
        self.SMTP_PWD = config["SMTP"]["SMTP_PWD"]
        self.SMTP_DEST = config["SMTP"]["SMTP_DEST"]

    # ---------------------------------------------------------------------------------------------
    def prepara(self, imputador:str, missing_rate:str, dataset:str):
        """
        Prepara o e-mail para ser enviado.
        """

        # Assunto e corpo do e-mail
        assunto = "Código da imputação"
        corpo = f"""
        <html>
        <body>
            <p>Código para {imputador.upper()} no dataset {dataset} com MD = {missing_rate} % começou!</p>
        </body>
        </html>
        """

        # Criar uma mensagem de e-mail
        mensagem = MIMEMultipart()
        mensagem["From"] = self.SMTP_USER
        mensagem["To"] = self.SMTP_DEST
        mensagem["Subject"] = assunto
        mensagem.attach(MIMEText(corpo, "html"))

        return mensagem

    # ---------------------------------------------------------------------------------------------
    def prepara_error(self, imputador:str, missing_rate:str, dataset:str, error:str):
        """
        Prepara o e-mail para ser enviado.
        """

        # Assunto e corpo do e-mail
        assunto = "Erro durante a execução do código"
        corpo = f"""
        <html>
        <body>
            <p>Código para {imputador.upper()} no dataset {dataset} com MD = {missing_rate} % deu erro!</p>
            <p> Erro: {error} </p>
        </body>
        </html>
        """

        # Criar uma mensagem de e-mail
        mensagem = MIMEMultipart()
        mensagem["From"] = self.SMTP_USER
        mensagem["To"] = self.SMTP_DEST
        mensagem["Subject"] = assunto
        mensagem.attach(MIMEText(corpo, "html"))

        return mensagem
    # ---------------------------------------------------------------------------------------------
    def envia(self, msg):
        """
        Envia e-mail

        Args:
        msg: email.mime.multipart.MIMEMultipart
        """
        
        server = smtplib.SMTP(self.SMTP_SERVER,self. SMTP_PORT)
        server.starttls()
        server.login(self.SMTP_USER, self.SMTP_PWD)
        server.sendmail(self.SMTP_USER, self.SMTP_DEST, msg.as_string())
        server.quit()
        