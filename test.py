#!/usr/bin/env python3
# coding: utf-8


import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject
import pandas as pd
from LogRegressionDetailed_24Bus_24Period00 import *
import numpy as np
import os
import time
import psutil

class TrainThread(GObject.GObject):
    __gsignals__ = {
        'log_signal': (GObject.SIGNAL_RUN_FIRST, None, (str,))
    }
    
    def __init__(self, train_file, test_file, num_epochs, batch_size):
        GObject.GObject.__init__(self)
        self.train_file = train_file
        self.test_file = test_file
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_path = "models/power_system_transformer_model.pth"
        self.input_dim = 576
        self.output_dim = 792
    
    def run(self):
        try:
            self.model_path, self.input_dim, self.output_dim = train_main(self.train_file, self.test_file, self.num_epochs, self.batch_size, log_callback=self.emit_log)
        except Exception as e:
            print(f"训练错误: {str(e)}")
            self.emit_log(f"训练错误: {str(e)}")
            
    def emit_log(self, message):
        self.emit('log_signal', message)

class MyWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Power System Transformer ID:19311")
        self.initUI()

    def initUI(self):
        # 主水平布局
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.add(main_box)
        
        # 左侧文本显示框
        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        main_box.pack_start(left_box, True, True, 0)
        
        self.text_display = Gtk.TextView()
        self.text_display.set_editable(False)
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.add(self.text_display)
        left_box.pack_start(scrolled_window, True, True, 0)
        # 右侧按钮区域
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        main_box.pack_start(right_box, False, False, 0)
        
        # 训练数据按钮
        btn_train_data = Gtk.Button(label='选择训练数据')
        btn_train_data.connect('clicked', self.open_train_file)
        right_box.pack_start(btn_train_data, False, False, 0)
        
        # 验证数据按钮
        btn_test_data = Gtk.Button(label='选择验证数据')
        btn_test_data.connect('clicked', self.open_test_file)
        right_box.pack_start(btn_test_data, False, False, 0)
        
        # 参数输入框
        self.batch_size_input = Gtk.Entry(placeholder_text='输入batch size (默认:64)')
        right_box.pack_start(self.batch_size_input, False, False, 0)
        
        self.epoch_input = Gtk.Entry(placeholder_text='输入epoch数 (默认:50)')
        right_box.pack_start(self.epoch_input, False, False, 0)
        
        # 训练按钮
        btn_train = Gtk.Button(label='开始训练')
        btn_train.connect('clicked', self.train)
        right_box.pack_start(btn_train, False, False, 0)
        
        # 加载模型按钮
        btn_load = Gtk.Button(label='加载模型')
        btn_load.connect('clicked', self.load_model)
        right_box.pack_start(btn_load, False, False, 0)
        
        # 输入数据按钮
        btn_input_data = Gtk.Button(label='选择输入数据')
        btn_input_data.connect('clicked', self.open_data_file)
        right_box.pack_start(btn_input_data, False, False, 0)
        
        # 预测按钮
        btn_predict = Gtk.Button(label='运行预测')
        btn_predict.connect('clicked', self.predict)
        right_box.pack_start(btn_predict, False, False, 0)
        
        # CPU占用显示
        self.cpu_label = Gtk.Label(label="")
        right_box.pack_start(self.cpu_label, False, False, 0)
        
        self.set_default_size(800, 500)
    
    
    def open_train_file(self, widget):
        dialog = Gtk.FileChooserDialog(
            title='打开训练数据文件',
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            '取消', Gtk.ResponseType.CANCEL,
            '选择', Gtk.ResponseType.OK
        )
        
        filter_txt = Gtk.FileFilter()
        filter_txt.set_name('文本文件')
        filter_txt.add_pattern('*.txt')
        dialog.add_filter(filter_txt)
        
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.train_file = dialog.get_filename()
            self.append_log(f"成功打开训练数据文件: {self.train_file}")
        dialog.destroy()

    def open_test_file(self, widget):
        dialog = Gtk.FileChooserDialog(
            title='打开测试数据文件',
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
            buttons=(
                '取消', Gtk.ResponseType.CANCEL,
                '选择', Gtk.ResponseType.OK
            )
        )
        filter_txt = Gtk.FileFilter()
        filter_txt.set_name('文本文件')
        filter_txt.add_pattern('*.txt')
        dialog.add_filter(filter_txt)
        
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.test_file = dialog.get_filename()
            self.append_log(f"成功打开测试数据文件: {self.test_file}")
        dialog.destroy()

    def train(self, widget):
        if hasattr(self, 'train_file') and hasattr(self, 'test_file'):
            try:
                batch_size = int(self.batch_size_input.get_text()) if self.batch_size_input.get_text() else 64
                epoch = int(self.epoch_input.get_text()) if self.epoch_input.get_text() else 50
            except ValueError:
                self.append_log("请输入有效的数字参数")
                return
                
            self.append_log(f"正在训练，batch size: {batch_size}, epoch: {epoch}")
            self.train_thread = TrainThread(self.train_file, self.test_file, epoch, batch_size)
            self.train_thread.connect('log_signal', self.append_log)
            self.train_thread.run()
        else:
            self.append_log("请先选择训练数据文件和测试数据文件")

    def load_model(self, widget):
        default_model_path = "models/power_system_transformer_model.pth"
        default_input_dim = 576
        default_output_dim = 792
        
        try:
            if hasattr(self, 'train_thread') and hasattr(self.train_thread, 'model_path'):
                self.append_log("正在加载训练后的模型...")
                self.model = load_pretrained_model(self.train_thread.model_path,
                                                self.train_thread.input_dim,
                                                self.train_thread.output_dim)
                self.append_log("训练后的模型加载完成")
            else:
                self.append_log("正在加载默认模型...")
                self.model = load_pretrained_model(default_model_path,
                                                default_input_dim,
                                                default_output_dim)
                self.append_log("默认模型加载完成")
        except Exception as e:
            self.append_log(f"模型加载失败: {str(e)}")

    def open_data_file(self, widget):
        dialog = Gtk.FileChooserDialog(
            title='打开输入数据文件',
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
            buttons=(
                '取消', Gtk.ResponseType.CANCEL,
                '选择', Gtk.ResponseType.OK
            )
        )
        filter_txt = Gtk.FileFilter()
        filter_txt.set_name('文本文件')
        filter_txt.add_pattern('*.txt')
        dialog.add_filter(filter_txt)
        
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.data_file = dialog.get_filename()
            self.append_log(f"成功打开输入数据文件: {self.data_file}")
        dialog.destroy()

    def predict(self, widget):
        if hasattr(self, 'model') and hasattr(self, 'data_file'):
            try:
                # 开始CPU监控
                self.cpu_monitor = True
                GObject.timeout_add(1000, self.update_cpu_usage)
                
                start_time = time.time()
                predictions = predict_from_file(self.model, self.data_file)
                end_time = time.time()
                time_consumed = end_time - start_time
                self.append_log(f"预测结果: {predictions}")
                self.append_log(f"预测耗时: {time_consumed:.4f} 秒")
                
                # 保存预测结果为txt文件
                output_file = os.path.splitext(self.data_file)[0] + "_predictions.txt"
                np.savetxt(output_file, predictions, fmt='%d')
                self.append_log(f"预测结果已保存到: {output_file}")
                
                # 停止CPU监控
                self.cpu_monitor = False
            except Exception as e:
                self.append_log(f"预测过程出现错误: {str(e)}")
                self.cpu_monitor = False
        else:
            self.append_log("请先加载模型和选择输入数据文件")
            
    def update_cpu_usage(self):
        if hasattr(self, 'cpu_monitor') and self.cpu_monitor:
            psutil.cpu_percent(interval=None)  # 初始化
            while True:
                cpu_percent = psutil.cpu_percent(interval=1)
                print(f"CPU使用率: {cpu_percent}%")
                self.cpu_label.set_text(f"CPU占用: {cpu_percent}%")
                time.sleep(1)  # 适当延迟
            
            return True
        return False
                
    def on_train_finished(self, widget=None):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="训练完成！"
        )
        dialog.run()
        dialog.destroy()

    def append_log(self, widget, message=None):
        if message is None:  # 处理直接调用的情况
            message = widget
        buffer = self.text_display.get_buffer()
        buffer.insert_at_cursor(message + "\n")

if __name__ == '__main__':
    win = MyWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()