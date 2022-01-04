<?php
header("Content-Type: text/html; charset=UTF-8");

require_once "naverBookApi.php";

$book = new Book();

echo $book->getBookSearch($_POST['query']);
?>