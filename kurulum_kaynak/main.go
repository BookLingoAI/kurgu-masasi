// Kurgu Masası kurulum programı: gömülü dosyaları geçici bir klasöre açar ve kurulum sihirbazını
// (Windows PowerShell + Windows Forms) başlatır. Sihirbaz kapanınca geçici dosyaları siler.
package main

import (
	"archive/zip"
	"bytes"
	_ "embed"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

//go:embed payload.zip
var payload []byte

func messageBox(text string, flags uintptr) {
	user32 := syscall.NewLazyDLL("user32.dll")
	proc := user32.NewProc("MessageBoxW")
	t, _ := syscall.UTF16PtrFromString(text)
	c, _ := syscall.UTF16PtrFromString("Kurgu Masası Kurulum")
	proc.Call(0, uintptr(unsafe.Pointer(t)), uintptr(unsafe.Pointer(c)), flags)
}

func extract(dst string) error {
	r, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
	if err != nil {
		return err
	}
	for _, f := range r.File {
		p := filepath.Join(dst, filepath.FromSlash(f.Name))
		if !strings.HasPrefix(p, filepath.Clean(dst)+string(os.PathSeparator)) {
			return fmt.Errorf("geçersiz dosya yolu: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(p)
		if err != nil {
			rc.Close()
			return err
		}
		_, err = io.Copy(out, rc)
		rc.Close()
		out.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func main() {
	tmp, err := os.MkdirTemp("", "KurguMasasiKurulum-")
	if err != nil {
		messageBox("Geçici klasör oluşturulamadı: "+err.Error(), 0x10)
		os.Exit(1)
	}
	defer os.RemoveAll(tmp)
	if err := extract(tmp); err != nil {
		messageBox("Kurulum dosyaları açılamadı: "+err.Error(), 0x10)
		os.Exit(1)
	}
	ps := filepath.Join(os.Getenv("SystemRoot"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
	if _, err := os.Stat(ps); err != nil {
		ps = "powershell.exe"
	}
	cmd := exec.Command(ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA", "-WindowStyle", "Hidden",
		"-File", filepath.Join(tmp, "kurulum.ps1"), "-Kaynak", tmp)
	cmd.Dir = tmp
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() != 0 {
			messageBox(fmt.Sprintf("Kurulum penceresi beklenmedik şekilde kapandı (kod %d).\n\nYedek yol: zip içindeki "+
				"'Yedek kurulum' klasöründe KUR.bat dosyasını çalıştırın.", ee.ExitCode()), 0x30)
			os.Exit(1)
		}
		if _, ok := err.(*exec.ExitError); !ok {
			messageBox("PowerShell başlatılamadı: "+err.Error(), 0x10)
			os.Exit(1)
		}
	}
}
