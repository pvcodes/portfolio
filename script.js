setTimeout(() => {
    var el = document.getElementById("presentation");
    var cursore = document.getElementById("cursor");
    cursore.style.visibility = "hidden";
    el.style.visibility = "visible";
}, 3000);

window.onload = () => {
    setTimeout(
        () => {
            document.getElementById('popup').style.opacity = 0;
        }, 5000);
};