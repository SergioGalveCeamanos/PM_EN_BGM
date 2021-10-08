# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:52:36 2021

@author: sega01
"""
from fpdf import FPDF
import datetime

class PDF(FPDF):
    def __init__(self,tit='',fon='Brandom_',da=''):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.Header_height = 25
        self.Bottom_height = 25
        self.Lateral_borders= 10
        self.set_auto_page_break(True)
        self.add_font('Brandom_Title', '', '/pm_manager/classes/BrandonGrotesqueOffice-Medium.ttf', True)
        self.add_font('Brandom_Text', '', '/pm_manager/classes/BrandonGrotesqueOffice-Light.ttf', True)
        self.fon_title=fon+'Title'
        self.fon_text=fon+'Text'
        self.set_font('Arial', 'B', 14)
        if da=='':
            self.date_creation=datetime.datetime.now().ctime()
        self.set_title(tit+self.date_creation)
        
    def header(self):
        # Custom logo and positioning
        # Create an `assets` folder and put any wide and short image inside
        # Name the image `logo.png`
        self.image("/pm_manager/classes/Logo_Lauda.png", 5, 5, w=(self.Header_height-15)*5.94/1.2, h=self.Header_height-15)
        self.set_font(self.fon_title, '', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 0, 'Fault Detection Report', 0, 0, 'R')
        self.ln(self.Header_height)
        
    def footer(self):
        # Page numbers in the footer
        self.image("/pm_manager/classes/lema_lauda.jpg", 5, self.HEIGHT-10, w=2*4.19/0.18, h=2)
        self.set_y(-25)
        self.set_font(self.fon_text, '', 8)
        self.set_text_color(128)
        self.cell(0, self.Bottom_height, 'Page ' + str(self.page_no()), 0, 0, 'C')
        self.cell(0, self.Bottom_height, self.date_creation, 0, 0, 'R')
        

    def page_body(self, images, text_space=0):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        d=self.Header_height+text_space
        img_sp=self.HEIGHT-(2*self.Header_height+text_space)
        for i in range(len(images)):
            self.image(images[0], self.Lateral_borders, d + i*img_sp / len(images), w=self.WIDTH - 2*self.Lateral_borders , h=(img_sp / len(images))-5)
    
    def chapter_title(self, sect ):
        # Arial 12
        self.set_font(self.fon_title, '', 16)
        self.set_line_width(0.3)
        # Background color
        #self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 10, sect , 0, 1, 'L', 0)
        # Line break
        self.ln(5)

    def chapter_body(self, text):
        # Times 12
        self.set_font(self.fon_text, '', 10)
        self.set_line_width(0.1)
        # Output justified text
        self.multi_cell(0, 10, text, 0, 1, 'L', 0)
        # Line break
        self.ln(15)       

    def print_chapter(self, title, name):
        self.chapter_title(title)
        self.chapter_body(name)
        
    def print_page(self, images=[],text=[],title=''):
        # Generates the report
        self.add_page()
        if text!=[]:
            self.print_chapter(title, text)
        if images!=[]: 
            self.page_body(images, text_space=60)
 
        
#pdf = PDF(tit='Fault Detection Report')
#images=[[r"C:\Users\sega01\Downloads\Policy_Management.png",r"C:\Users\sega01\Downloads\Blank diagram.png"]]
#title='Test Doc'
#text='It Is The Small Things, Everyday Deeds Of Ordinary Folk That Keeps The Darkness At Bay. Simple Acts Of Love And Kindness.'
#.print_page(images[0],text,title)

#pdf.output(r"C:\Users\sega01\Downloads\Policy_Management.pdf", 'F')